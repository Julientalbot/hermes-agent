"""Exercise the real CDP transport without launching or navigating a browser."""
import json
import threading

from tools.bot_desktop import browser, lease, runtime


def test_present_existing_selected_tab_without_changing_geometry(tmp_path, monkeypatch):
    from websockets.sync.server import serve
    commands = []

    def handler(ws):
        for raw in ws:
            msg = json.loads(raw)
            method, params = msg['method'], msg.get('params', {})
            commands.append((method, params))
            result = {}
            if method == 'Target.getTargets':
                result = {'targetInfos': [{'type': 'page', 'targetId': 'background'}, {'type': 'page', 'targetId': 'selected'}]}
            elif method == 'Target.attachToTarget':
                result = {'sessionId': params['targetId']}
            elif method == 'Runtime.evaluate':
                result = {'result': {'value': msg['sessionId'] == 'selected'}}
            elif method == 'Browser.getWindowForTarget':
                assert params['targetId'] == 'selected'
                result = {'windowId': 7}
            ws.send(json.dumps({'id': msg['id'], 'result': result}))

    with serve(handler, '127.0.0.1', 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        (tmp_path / 'DevToolsActivePort').write_text(f'{port}\n/devtools/browser/test-id\n')
        monkeypatch.setattr(browser, 'profile_dir', lambda: tmp_path)
        monkeypatch.setattr(browser, 'running_instance_cdp_port', lambda profile: port)
        assert browser.present_running_browser() == {'success': True}
        server.shutdown()
        thread.join(2)
    assert commands[-2:] == [('Browser.setWindowBounds', {'windowId': 7, 'bounds': {'windowState': 'maximized'}}),
                              ('Target.activateTarget', {'targetId': 'selected'})]
    assert not any(method in {'Target.createTarget', 'Page.navigate', 'Browser.close'} for method, _ in commands)


def test_missing_browser_and_human_control_leave_browser_untouched(tmp_path, monkeypatch):
    from tools.browser_tool_session import run_fenced
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(browser, 'profile_dir', lambda: tmp_path)
    assert not browser.present_running_browser()['success']
    monkeypatch.setattr(runtime, 'published_env', lambda: {'DISPLAY': ':99'})
    held = lease.acquire('human-test')
    def forbidden():
        raise AssertionError('must not contact the browser during human control')
    result = run_fenced({'features': {'local': True}}, forbidden)
    assert result['code'] == 'human_has_control'
    assert lease.get().epoch == held.epoch
