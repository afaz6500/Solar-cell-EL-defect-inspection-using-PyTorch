"""HTTP contract tests for the local beta; no inference or dataset changes."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
import web_app as web


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(web, "JOBS", Path(self.temp.name))
        self.state_patch = patch.object(web, "STATE", {})
        self.root_patch.start()
        self.state_patch.start()
        self.server = web.ThreadingHTTPServer(("127.0.0.1", 0), web.Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.state_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def request(self, path, method="GET", data=None, headers=None):
        if isinstance(data, dict):
            data = json.dumps(data).encode()
        req = Request(self.url + path, data=data, method=method, headers=headers or {})
        try:
            response = urlopen(req)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, response.read()

    def create(self, **config):
        code, body = self.request('/api/jobs', 'POST', config)
        self.assertEqual(code, 201)
        return json.loads(body)['id']

    def upload(self, job_id, name="panel.png"):
        buffer = io.BytesIO()
        Image.new('RGB', (240, 120), '#858585').save(buffer, format='PNG')
        return self.request(f'/api/jobs/{job_id}/files?name={name}', 'PUT', buffer.getvalue())

    def test_upload_export_and_delete_are_job_scoped(self):
        job_id = self.create(mode='panel', rows=6, cols=12)
        self.assertEqual(self.upload(job_id)[0], 201)
        self.assertEqual(self.upload(job_id)[0], 201)
        job = web.STATE[job_id]
        self.assertEqual(job['total'], 144)
        self.assertNotEqual(job['sources'][0]['path'], job['sources'][1]['path'])
        self.assertEqual(self.request(f'/api/jobs/{job_id}/asset?path=../../README.md')[0], 404)
        job['results'] = [{'source': '=SUM(A1)', 'prediction': 'normal'}]
        code, body = self.request(f'/api/jobs/{job_id}/export?format=csv')
        self.assertEqual(code, 200)
        self.assertIn("'=SUM(A1)", body.decode('utf-8-sig'))
        self.assertEqual(self.request(f'/api/jobs/{job_id}', 'DELETE')[0], 200)
        self.assertFalse((web.JOBS / job_id).exists())

    def test_invalid_settings_images_and_origin(self):
        for cfg in ({'rows': 0}, {'gap': -1}, {'rows': 2.5}, {'margin': 'nan'}, {'mode':'invalid'}):
            self.assertEqual(self.request('/api/jobs','POST',cfg)[0], 400)
        job_id = self.create()
        self.assertEqual(self.request(f'/api/jobs/{job_id}/files?name=x.png','PUT',b'not an image')[0], 400)
        self.assertEqual(self.request('/api/jobs','POST',{}, {'Origin':'https://example.com'})[0], 403)
        self.assertEqual(self.request('/api/status',headers={'Host':'example.com'})[0], 403)
        self.assertEqual(self.request(f'/api/jobs/{job_id}/start','POST')[0], 400)

    def test_retry_rejected_preserves_results_and_evidence(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock
        job_id = self.create()
        self.upload(job_id)
        job = web.STATE[job_id]
        source = job['sources'][0]
        rejected = dict(index=0, source=source['name'], source_id=source['id'], image=source['path'],
                        row=None, col=None, bbox=None, prediction='REJECTED', ood_reason='Too dark')
        accepted = dict(index=1, prediction='normal', image=source['path'])
        job.update(status='completed', results=[rejected,accepted], done=2, total=2)
        with patch.object(web.POOL,'submit') as submit, patch.object(web,'model_info',return_value={'ready':True}):
            self.assertEqual(self.request(f'/api/jobs/{job_id}/retry-rejected','POST')[0],200)
            submit.assert_called_once_with(web.retry_rejected,job_id)
            self.assertEqual(self.request(f'/api/jobs/{job_id}/retry-rejected','POST')[0],400)
        predict = Mock(return_value=dict(prediction='UNCERTAIN',confidence=70,dsi=40,probabilities={'crack':0.7},ood_reason='Too dark'))
        analyse = Mock()
        with patch.dict(sys.modules, {'scripts.predict':SimpleNamespace(predict_image=predict), 'scripts.analyse':SimpleNamespace(analyse_image=analyse)}), patch.object(web,'engine',return_value=(None,['crack'],{},None)):
            web.retry_rejected(job_id)
        self.assertTrue(predict.call_args.kwargs['force_process'])
        self.assertEqual(predict.call_count,1)
        self.assertEqual(job['results'][1],accepted)
        self.assertEqual(job['results'][0]['original_ood_reason'],'Too dark')
        self.assertTrue(job['results'][0]['ood_override'])
        self.assertEqual(job['results'][0]['likely_class'],'crack')
        self.assertEqual(job['done'],2)
        self.assertEqual(job['status'],'completed')
        self.assertEqual(self.request(f'/api/jobs/{job_id}/retry-rejected','POST')[0],400)
        job['results'][0] = rejected
        job['settings']['visuals'] = True
        analyse.return_value = dict(error='Cannot decode image')
        with patch.dict(sys.modules, {'scripts.predict':SimpleNamespace(predict_image=predict), 'scripts.analyse':SimpleNamespace(analyse_image=analyse)}), patch.object(web,'engine',return_value=(None,['crack'],{},None)):
            web.retry_rejected(job_id)
        self.assertTrue(analyse.call_args.kwargs['force_process'])
        self.assertEqual(job['results'][0]['prediction'],'ERROR')
        self.assertIsNone(job['results'][0]['dsi'])

    def test_freeform_margins(self):
        from scripts.panel_splitter import split_panel_image
        margins = dict(left=20, top=10, right=40, bottom=30)
        job_id = self.create(mode='panel', rows=2, cols=3, margins=margins)
        self.assertEqual(self.upload(job_id)[0], 201)
        self.assertEqual(web.STATE[job_id]['settings']['margins'], margins)
        source = Path(self.temp.name) / 'source.png'
        Image.new('L', (240,120)).save(source)
        cells = split_panel_image(source, Path(self.temp.name)/'crops', 2, 3, margins=margins)
        self.assertEqual(cells[0].bbox, (20,10,80,50))
        self.assertEqual(cells[-1].bbox, (140,50,200,90))
        for bad in ({}, dict(margins, left=-1), dict(margins, top=1.5)):
            self.assertEqual(self.request('/api/jobs','POST',dict(mode='panel',margins=bad))[0],400)
        invalid = self.create(mode='panel', rows=2, cols=3, margins=dict(margins,left=220))
        self.assertEqual(self.upload(invalid)[0],400)
        with self.assertRaises(ValueError):
            split_panel_image(source, Path(self.temp.name)/'invalid', 2, 3, margins=dict(margins,left=220))

    def test_cancel_queue_and_restore(self):
        job_id=self.create()
        self.upload(job_id)
        with patch.object(web.POOL,'submit') as submit, patch.object(web,'model_info',return_value={'ready':True}):
            self.assertEqual(self.request(f'/api/jobs/{job_id}/start','POST')[0],200)
            submit.assert_called_once_with(web.process,job_id)
        self.assertEqual(self.request(f'/api/jobs/{job_id}','DELETE')[0],400)
        self.assertEqual(self.request(f'/api/jobs/{job_id}/cancel','POST')[0],200)
        self.assertEqual(web.STATE[job_id]['status'],'cancelled')
        second=self.create()
        web.restore()
        self.assertEqual(web.STATE[second]['status'],'interrupted')

    def test_relational_dashboard_profile_and_pdf(self):
        code, body = self.request('/api/profile', 'POST', {'name':'Aina', 'company':'Solar & Co'})
        self.assertEqual(code, 200)
        job_id = self.create(title='Rooftop A')
        self.upload(job_id)
        job = web.STATE[job_id]
        source = job['sources'][0]
        job.update(status='completed', done=1)
        job['results'] = [dict(index=0, source_id=source['id'], source=source['name'], prediction='normal',
                               confidence=95.2, dsi=2.1, image=source['path'])]
        web.save(job)
        web.save(job)
        code, body = self.request('/api/dashboard')
        summary = json.loads(body)
        self.assertEqual(summary['totals']['cells'], 1)
        self.assertEqual(summary['totals']['normal'], 1)
        self.assertEqual(summary['recent'][0]['title'], 'Rooftop A')
        self.assertEqual(sum(a['event']=='inspection_completed' for a in summary['activity']), 1)
        code, body = self.request(f'/api/jobs/{job_id}/export?format=pdf')
        self.assertEqual(code, 200)
        self.assertTrue(body.startswith(b'%PDF-'))
        web.STATE.clear()
        web.restore()
        self.assertEqual(web.STATE[job_id]['profile']['company'], 'Solar & Co')
        self.assertEqual(web.STATE[job_id]['results'][0]['prediction'], 'normal')
        self.assertEqual(self.request(f'/api/jobs/{job_id}/export?format=exe')[0],400)
        self.assertEqual(self.request(f'/api/jobs/{job_id}', 'DELETE')[0],200)
        with web.web_store.connection(web.JOBS) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM findings').fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM items').fetchone()[0],0)
            self.assertGreater(db.execute('SELECT COUNT(*) FROM activity_logs').fetchone()[0],0)

    def test_pdf_requires_results_and_invalid_profile_is_rejected(self):
        job_id=self.create()
        self.assertEqual(self.request(f'/api/jobs/{job_id}/export?format=pdf')[0],400)
        self.assertEqual(self.request('/api/profile','POST',{'name':''})[0],400)


if __name__ == '__main__':
    unittest.main()
