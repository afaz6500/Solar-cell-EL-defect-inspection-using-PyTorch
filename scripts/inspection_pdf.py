"""Paginated PDF inspection reports; consumes saved results, never reruns inference."""
from collections import Counter
from datetime import datetime, timezone
import io
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab import __file__ as reportlab_file
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether


def report_pdf(job, folder):
    folder = Path(folder).resolve()
    if 'SolarSans' not in pdfmetrics.getRegisteredFontNames():
        fonts = Path(reportlab_file).parent / 'fonts'
        pdfmetrics.registerFont(TTFont('SolarSans', str(fonts / 'Vera.ttf')))
        pdfmetrics.registerFont(TTFont('SolarBold', str(fonts / 'VeraBd.ttf')))
    ink, teal, gray = colors.HexColor('#172d35'), colors.HexColor('#087d68'), colors.HexColor('#667782')
    styles = getSampleStyleSheet()
    for name in ['Normal', 'BodyText', 'Heading1', 'Heading2', 'Heading3']:
        styles[name].fontName = 'SolarSans'
        styles[name].textColor = ink
    styles.add(ParagraphStyle('TitleSolar', fontName='SolarBold', fontSize=27, leading=34, textColor=ink, spaceAfter=12))
    styles.add(ParagraphStyle('SmallSolar', fontName='SolarSans', fontSize=8, leading=12, textColor=gray, spaceAfter=5))
    styles.add(ParagraphStyle('CellSolar', fontName='SolarSans', fontSize=8, leading=12, textColor=ink))
    styles.add(ParagraphStyle('WhiteSolar', fontName='SolarBold', fontSize=8, leading=12, textColor=colors.white))
    styles.add(ParagraphStyle('MetricSolar', fontName='SolarBold', fontSize=21, leading=27, alignment=TA_CENTER, textColor=teal))
    styles.add(ParagraphStyle('LabelSolar', fontName='SolarSans', fontSize=8, leading=12, alignment=TA_CENTER, textColor=gray))
    p = lambda value, style='CellSolar': Paragraph(escape(str(value)), styles[style])
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=42, rightMargin=42, topMargin=64, bottomMargin=52,
                            title='Solar Inspection - ' + job['id'][:8], author='Solar Inspection')
    width = A4[0] - 84
    results = job['results']
    counts = Counter(r.get('prediction', 'ERROR') for r in results)
    review = sum(counts[k] for k in ('UNCERTAIN','REJECTED','ERROR'))
    defects = len(results) - review - counts['normal']
    title = job.get('title') or (job['sources'][0]['name'] if job['sources'] else 'Inspection')
    profile = job.get('profile', {})
    generated = datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')

    def page(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(teal)
        canvas.setLineWidth(2)
        canvas.line(42, A4[1]-35, A4[0]-42, A4[1]-35)
        canvas.setFont('SolarBold', 9)
        canvas.setFillColor(ink)
        canvas.drawString(42, A4[1]-25, 'SOLAR INSPECTION')
        canvas.setFont('SolarSans', 8)
        canvas.setFillColor(gray)
        canvas.drawRightString(A4[0]-42, A4[1]-25, 'INSPECTION REPORT')
        canvas.line(42, 38, A4[0]-42, 38)
        canvas.drawString(42, 25, f"{job['id'][:8].upper()} | Model-assisted findings - engineer review required")
        canvas.drawRightString(A4[0]-42, 25, f'Page {document.page}')
        canvas.restoreState()

    story = [p('Solar cell inspection', 'TitleSolar'), p(title, 'Heading2'), Spacer(1, 10)]
    status = job['status'].upper()
    if status != 'COMPLETED':
        status += ' - PARTIAL RESULTS'
    metadata = [
        ['Report ID', job['id'][:8].upper(), 'Status', status],
        ['Business', profile.get('company') or 'Not specified', 'Operator', profile.get('name') or 'Local operator'],
        ['Started (UTC)', job['created'][:19].replace('T',' '), 'Generated', generated],
        ['Input', 'Whole panels' if job['settings']['mode']=='panel' else 'Individual cells', 'Coverage', f"{job['done']} / {job['total']} cells"],
    ]
    table = Table([[p(v) for v in row] for row in metadata], colWidths=[75,180,65,width-320])
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f3f6f7')),
                               ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story += [table, Spacer(1,20)]
    metric = Table([[p(n,'MetricSolar') for n in (len(results),defects,counts['normal'],review)],
                    [p(v,'LabelSolar') for v in ('Cells processed','Defect predictions','Normal predictions','Uncertain / invalid')]], colWidths=[width/4]*4)
    story += [metric, Spacer(1,18), p('Inspection summary','Heading2')]
    story.append(p('These are model predictions, not confirmed defects. Confidence is calibrated; DSI is an indicative severity score, not measured power loss. Uncertain, rejected, and error results need manual review.','BodyText'))
    cfg = job['settings']
    if cfg['mode']=='panel':
        margin_label = (", ".join(f"{side} {value}px" for side,value in cfg["margins"].items())
                        if "margins" in cfg else f"{cfg['margin']}%")
        story += [Spacer(1,8),p(f"Grid: {cfg['rows']} rows x {cfg['cols']} columns; outer margin {margin_label}; cell gap {cfg['gap']}%.",'SmallSolar')]
    distribution = ', '.join(f"{name.replace('_',' ').title()}: {count}" for name,count in counts.most_common())
    story += [Spacer(1,8),p(distribution,'BodyText')]
    if job.get('errors'):
        story += [Spacer(1,8),p('Source errors','Heading3')]
        story += [p(f"{e['source']}: {e['message']}") for e in job['errors']]
    if job['sources']:
        source = job['sources'][0]
        path = (folder / source['path']).resolve()
        if path.is_relative_to(folder) and path.is_file():
            image = Image(str(path))
            image._restrictSize(width, 155)
            story += [Spacer(1,18),KeepTogether([p('Source preview - ' + source['name'],'Heading3'), image])]
    story += [Spacer(1,14),p('Scope: 10 supported cell conditions. Fragment, printing error, and vertical dislocation are excluded. This report is not a safety certificate or a verified electrical-performance assessment.','SmallSolar')]
    story += [PageBreak(),p('Cell findings','Heading1'),p('Every processed cell is included below. A dash means the value is unavailable.','SmallSolar'),Spacer(1,8)]
    header = ['Source / location','Prediction','Confidence','DSI / 100','Review note']
    rows = [[p(v,'WhiteSolar') for v in header]]
    for result in results:
        source = result.get('source','Image') + (f" / R{result['row']} C{result['col']}" if result.get('row') else '')
        prediction = result.get('prediction','ERROR').replace('_',' ').title()
        if result.get('prediction')=='UNCERTAIN' and result.get('likely_class'):
            prediction += ' (leading: ' + result['likely_class'].replace('_',' ') + ')'
        note = result.get('error') or (result.get('ood_reason','Review required') if result.get('ood_override') or result.get('prediction') in ('UNCERTAIN','REJECTED','ERROR') else 'Visual review required')
        confidence = '-' if result.get('confidence') is None else f"{result['confidence']:.1f}%"
        dsi = '-' if result.get('dsi') is None else f"{result['dsi']:.1f}"
        rows.append([p(v) for v in (source,prediction,confidence,dsi,note)])
    findings = Table(rows, colWidths=[110,106,63,58,width-337], repeatRows=1, hAlign='LEFT')
    findings.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),teal),('VALIGN',(0,0),(-1,-1),'TOP'),
                                 ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f6f7')]),
                                 ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                                 ('LINEBELOW',(0,0),(-1,0),.5,teal)]))
    story += [findings,Spacer(1,18),p('Model and calibration','Heading2'),p(job.get('model','10-class model'),'SmallSolar'),
              p('Checkpoint SHA-256: '+job.get('model_sha256','Not recorded'),'SmallSolar')]
    calibration = job.get('calibration',{})
    if calibration:
        story.append(p(f"Temperature: {calibration.get('temperature','-')} | Minimum confidence: {calibration.get('min_confidence','-')} | Maximum entropy: {calibration.get('max_entropy','-')}",'SmallSolar'))
    story += [PageBreak(), p('Visual analysis', 'Heading1'),
              p('Original / Grad-CAM / damage mask / skeleton / localization. These supporting visualizations are not verified defect boundaries. Refer to Cell findings for confidence and review notes.', 'SmallSolar')]
    if not job['settings'].get('visuals') and not any(r.get('visual') for r in results):
        story.append(p('Visual analysis was not enabled for this inspection.', 'BodyText'))
    else:
        for number, result in enumerate(results, 1):
            location = result.get('source', 'Image')
            if result.get('row'):
                location += f" / R{result['row']} C{result['col']}"
            caption = f"Cell {number}: {location} - {result.get('prediction', 'ERROR').replace('_', ' ').title()}"
            block = [p(caption, 'Heading3')]
            visual = result.get('visual')
            path = (folder / visual).resolve() if visual else None
            if path and path.is_relative_to(folder) and path.is_file():
                try:
                    visual_image = Image(str(path), lazy=0)
                    visual_image._restrictSize(width - 12, 170)
                    block.append(visual_image)
                except (OSError, ValueError):
                    block.append(p('The saved visual output could not be read.', 'SmallSolar'))
            else:
                block.append(p('No saved visual output is available for this cell.', 'SmallSolar'))
            block.append(Spacer(1, 14))
            story.append(KeepTogether(block))
    doc.build(story,onFirstPage=page,onLaterPages=page)
    return buffer.getvalue()
