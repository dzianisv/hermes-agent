"""Contract tests against the actual webhook verifier, never production DB."""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from gateway.platforms.webhook import WebhookAdapter

spec = importlib.util.spec_from_file_location('wake_under_test', Path(__file__).with_name('__init__.py'))
wake = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wake)

class WakeTests(unittest.TestCase):
    def setUp(self):
        wake._last.clear()

    def request(self):
        seen = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"status":"accepted"}'
        def post(req, **kw):
            seen.append(req)
            return Response()
        with patch.object(wake, '_secret', return_value='unit-secret'), patch.object(wake, '_cfg', return_value={}), patch.object(wake.urllib.request, 'urlopen', side_effect=post):
            wake._post('event data', 't_fixture', 'blocked')
        req = seen[0]
        headers = {k.lower():v for k,v in req.header_items()}
        class Headers(dict):
            def get(self,k,d=None): return super().get(k.lower(),d)
        return SimpleNamespace(headers=Headers(headers),match_info={'route_name':'kanban-wake'}),req.data

    def test_actual_receiver_accepts_emitter(self):
        req, body = self.request()
        self.assertTrue(WebhookAdapter._validate_signature(object(), req, body, 'unit-secret'))

    def test_tamper_rejected(self):
        req, body = self.request()
        self.assertFalse(WebhookAdapter._validate_signature(object(), req, body+b' ', 'unit-secret'))

    def test_request_has_idempotency_id(self):
        req, _ = self.request()
        self.assertTrue(req.headers.get('X-Request-ID'))

    def test_failed_send_can_retry_immediately(self):
        with patch.object(wake,'_cfg',return_value={}), patch.object(wake,'_post',side_effect=[OSError('offline'),None]) as post:
            for _ in range(2): wake._emit('blocked','t_fixture',title='fixture',reason='reason')
            self.assertEqual(post.call_count,2)

    def test_success_debounced(self):
        with patch.object(wake,'_cfg',return_value={}), patch.object(wake,'_post') as post:
            for _ in range(2): wake._emit('blocked','t_fixture',title='fixture',reason='reason')
            self.assertEqual(post.call_count,1)

    # ---------------------------------------------------------- redirector ---
    # Routing contract: the event goes to the card's stored origin, never "most recent session".

    def _run(self, subs, inject_ok=True):
        calls = []
        gw = SimpleNamespace(
            _schedule_plugin_message_injection=lambda **kw: (calls.append(kw), inject_ok)[1],
            _session_key_for_source=lambda src: f"key:{src.platform.value}:{src.chat_id}:{src.thread_id}")
        ev = SimpleNamespace(source=SimpleNamespace(platform='webhook', user_id='webhook:kanban-wake'),
                             text='KANBAN BLOCKED t_x', raw_message={'task_id': 't_x'})
        with patch.object(wake, '_origin_subs', return_value=subs):
            return wake._pre_gateway_dispatch(event=ev, gateway=gw), calls

    def test_notify_origin_gets_injected_into_its_thread(self):
        res, calls = self._run([{'platform': 'telegram', 'chat_id': '1', 'thread_id': '42', 'delivery_mode': 'notify'}])
        self.assertEqual(res['action'], 'skip')
        self.assertEqual([c['session_key'] for c in calls], ['key:telegram:1:42'])

    def test_wake_origin_left_to_core_notifier(self):
        res, calls = self._run([{'platform': 'telegram', 'chat_id': '1', 'thread_id': '42', 'delivery_mode': 'notify+wake'}])
        self.assertEqual(calls, [])
        self.assertIn('core notifier', res['reason'])

    def test_no_origin_sends_nothing_and_logs_info(self):
        with self.assertLogs(wake.logger, level='INFO') as log:
            res, calls = self._run([])
        self.assertEqual(calls, [])
        self.assertEqual(res['action'], 'skip')
        self.assertIn('no origin', '\n'.join(log.output))

    def test_non_wake_events_pass_through(self):
        ev = SimpleNamespace(source=SimpleNamespace(platform='telegram', user_id='1'), text='hi')
        self.assertIsNone(wake._pre_gateway_dispatch(event=ev, gateway=None))

    def test_real_session_key_routes_to_origin_thread(self):
        from gateway.session import build_session_key, SessionSource
        from gateway.config import Platform
        gw = SimpleNamespace(_session_key_for_source=lambda s: build_session_key(s))
        key = wake._origin_session_key(gw, {'platform': 'telegram', 'chat_id': '1916982742',
                                            'thread_id': '648763', 'chat_type': 'dm'})
        self.assertEqual(key, build_session_key(SessionSource(platform=Platform.TELEGRAM, chat_id='1916982742',
                                                              chat_type='dm', thread_id='648763')))
        self.assertTrue(key.endswith('648763'))

if __name__ == '__main__': unittest.main()
