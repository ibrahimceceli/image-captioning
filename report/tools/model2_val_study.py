"""Reproducible val-only preparation and review material; never trains a model."""
import argparse
import copy
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
STUDY = ROOT / 'results/model2_val_study_20261112'
SALT = 'model2-beam3-review-20261112-v1'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def selection():
    split_path = ROOT / 'src/data/artifacts/flickr8k_split.json'
    prediction_path = ROOT / 'results/model2_attention_lstm/val_captions_beam3.json'
    val = json.loads(split_path.read_text(encoding='utf-8'))['val']
    predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
    assert len(val) == 1000 and set(val) == set(predictions)
    rank = lambda image_id: hashlib.sha256(f'{SALT}:{image_id}'.encode()).hexdigest()
    ids = sorted(val, key=lambda image_id: (rank(image_id), image_id))[:40]
    write_json(STUDY / 'review/selection.json', {
        'split': 'val', 'population': 1000, 'sample_size': 40,
        'algorithm': 'ascending SHA256(UTF8(salt + colon + image_id)), tie-break image_id; first 40',
        'salt': SALT, 'quality_filter': False, 'split_sha256': digest(split_path),
        'beam3_predictions_sha256': digest(prediction_path),
        'image_ids': ids, 'ranks': {image_id: rank(image_id) for image_id in ids},
    })
    print('SELECTION_OK', len(ids), flush=True)


def acquire():
    url = 'https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k'
    archive = ROOT / 'data/downloads/flickr8k.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        partial = archive.with_suffix('.zip.part')
        request = urllib.request.Request(url, headers={'User-Agent': 'image-captioning-val-study/1.0'})
        count, started, last = 0, time.monotonic(), time.monotonic()
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
            print('HTTP', response.status, 'Content-Type', response.headers.get('Content-Type'), flush=True)
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
                count += len(block)
                if time.monotonic() - last > 10:
                    print('Downloaded MB', round(count / 1024**2), flush=True)
                    last = time.monotonic()
        if not zipfile.is_zipfile(partial):
            raise RuntimeError('Kaggle returned a non-ZIP response; account/browser access may be required')
        partial.rename(archive)
        print('DOWNLOAD_SECONDS', round(time.monotonic() - started, 1), flush=True)
    val = json.loads((ROOT / 'src/data/artifacts/flickr8k_split.json').read_text(encoding='utf-8'))['val']
    images = ROOT / 'data/flickr8k/Images'
    images.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        entries = {Path(item.filename).name: item for item in source.infolist() if not item.is_dir()}
        assert set(val).issubset(entries)
        for image_id in val:
            target = images / image_id
            if not target.exists():
                target.write_bytes(source.read(entries[image_id]))
        captions_path = ROOT / 'data/flickr8k/captions.txt'
        content = source.read(entries['captions.txt'])
        if captions_path.exists():
            assert captions_path.read_bytes() == content
        else:
            captions_path.write_bytes(content)
    expected = json.loads((ROOT / 'results/model2_attention_lstm/val_metrics_beam3.json').read_text())['reference_sha256']
    assert digest(captions_path) == expected, 'Reference source differs from historical beam3'
    write_json(STUDY / 'data_provenance.json', {
        'source': url, 'dataset': 'adityajn105/flickr8k', 'archive_sha256': digest(archive),
        'captions_sha256': digest(captions_path), 'matches_historical_beam3_references': True,
        'images_extracted': len(val), 'only_image_partition_extracted': 'val',
        'test_images_extracted': False, 'test_predictions_or_metrics': False,
    })
    print('VAL_DATA_READY', len(val), flush=True)


def review_material():
    from PIL import Image, ImageDraw, ImageFont
    from src.data.text import load_captions
    selection = json.loads((STUDY / 'review/selection.json').read_text(encoding='utf-8'))
    predictions = json.loads((ROOT / 'results/model2_attention_lstm/val_captions_beam3.json').read_text(encoding='utf-8'))
    refs = load_captions(ROOT / 'data/flickr8k/captions.txt')
    records = []
    folder = STUDY / 'review'
    for index, image_id in enumerate(selection['image_ids'], 1):
        assert len(refs[image_id]) == 5
        records.append({'number': index, 'image_id': image_id, 'beam3': predictions[image_id],
                        'references': refs[image_id], 'labels': None, 'reason': None,
                        'needs_user_review': None, 'visual_inspection': 'pending'})
    write_json(folder / 'records_pending.json', records)
    # Eight images per contact sheet, full uncropped image, linked to numbered records.
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    import textwrap
    contact_folder = ROOT / 'data/model2_review_20261112'
    contact_folder.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(records), 8):
        canvas = Image.new('RGB', (1600, 4 * 440), 'white')
        draw = ImageDraw.Draw(canvas)
        for slot, record in enumerate(records[offset:offset+8]):
            x, y = (slot % 2)*800, (slot // 2)*440
            with Image.open(ROOT / 'data/flickr8k/Images' / record['image_id']) as source:
                photo = source.convert('RGB')
                photo.thumbnail((780, 330))
                canvas.paste(photo, (x+(780-photo.width)//2, y))
            draw.text((x+10,y+332), f"{record['number']:02d} {record['image_id']}", fill='black',font=font)
            for line_no, line in enumerate(textwrap.wrap(record['beam3'], width=85)):
                draw.text((x+10,y+355+line_no*22), line, fill='black',font=font)
        canvas.save(contact_folder / f'contact_{offset//8+1:02d}.jpg', quality=94)
    lines = ['# Beam-3: 40 val görüntüsü — inceleme metinleri', '',
             'Seçim kaliteye göre filtrelenmedi. Görüntü numaraları contact_01–05 ile eşleşir.', '']
    for record in records:
        lines += [f"## {record['number']:02d} — {record['image_id']}", '',
                  f"Tahmin: {record['beam3']}", '', 'Referanslar:']
        lines += [f'- {text}' for text in record['references']]
        lines += ['']
    (folder / 'captions_and_references.md').write_text('\n'.join(lines), encoding='utf-8')
    print('REVIEW_MATERIAL_READY', len(records), flush=True)


def audit():
    import torch
    import torchvision
    import yaml
    def flatten(mapping, prefix=''):
        result = {}
        for key, value in mapping.items():
            path = f'{prefix}.{key}' if prefix else key
            result.update(flatten(value, path) if isinstance(value, dict) else {path: value})
        return result
    c1 = yaml.safe_load((ROOT/'configs/model1.yaml').read_text(encoding='utf-8'))
    c2 = yaml.safe_load((ROOT/'configs/model2.yaml').read_text(encoding='utf-8'))
    f1, f2 = flatten(c1), flatten(c2)
    diff = {key: {'model1': f1.get(key), 'model2': f2.get(key)}
            for key in sorted(f1.keys() | f2.keys()) if f1.get(key) != f2.get(key)}
    checkpoint = ROOT/'checkpoints/model2_attention_lstm_full_20261104_01/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['config'] == c2 and state['epoch'] == 10 and state['vocab_size'] == 2572
    val = json.loads((ROOT/'src/data/artifacts/flickr8k_split.json').read_text())['val']
    index = json.loads((ROOT/'data/features/model2_val_20261112/resnet50_layer4_index.json').read_text())
    assert len(index) == 1000 and set(index) == set(val)
    nucleus = {path.relative_to(ROOT).as_posix(): digest(path)
               for path in sorted((ROOT/'results/model2_attention_lstm').rglob('*'))
               if path.is_file() and 'nucleus' in path.as_posix()}
    write_json(STUDY/'audit.json', {
        'git_base': 'a625d64', 'checkpoint_sha256': digest(checkpoint),
        'checkpoint_epoch': state['epoch'], 'checkpoint_config_matches_model2_yaml': True,
        'checkpoint_historical_hash_available': False,
        'historical_checkpoint_identity_evidence': 'same path, epoch 10, architecture and vocab size; historical binary SHA256 was not recorded',
        'config_differences': diff,
        'same_sections': [key for key in c1.keys() & c2.keys() if c1[key] == c2[key]],
        'shared_base_sha256': digest(ROOT/'configs/base.yaml'),
        'split_sha256': digest(ROOT/'src/data/artifacts/flickr8k_split.json'),
        'vocab_sha256': digest(ROOT/'src/data/artifacts/vocab.json'),
        'validation_cache': {'image_count': 1000, 'exact_val_coverage': True,
                            'metadata': json.loads((ROOT/'data/features/model2_val_20261112/resnet50_layer4.meta.json').read_text())},
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'device': torch.cuda.get_device_name(0), 'nucleus_files_sha256': nucleus,
    })
    print('AUDIT_OK', json.dumps(diff), flush=True)


DECISION_FIELDS = ('labels', 'reason', 'uncertain_labels', 'needs_user_review',
                   'human_reviewed', 'review_status', 'uncertainty', 'human_labels', 'human_notes')
EXTRA_UNCERTAIN_CATEGORIES = ['kişi niteliği (boy/girl)']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reviewed_decision(row):
    """Compare semantics, while allowing absent optional empty fields."""
    return {key: row.get(key, [] if key in ('labels', 'uncertain_labels') else
                         False if key in ('human_reviewed', 'needs_user_review') else '')
            for key in DECISION_FIELDS}


def review_overview(summary):
    completed = len(summary['reviewed_numbers'])
    initial = len(summary['initial_pending_numbers'])
    pending = len(summary['needs_user_review'])
    progress = (f"Başlangıçta kullanıcı kontrolü bekleyen **{completed}/{initial} kayıt tamamlandı**; bekleyen kayıt yok."
                if not pending else f"Kullanıcı kararı kaydedilen {completed} kayıt; kontrol bekleyen {pending} kayıt var.")
    return (f"{progress} {summary['n']} görüntünün **{summary['any_label']}** tanesinde en az bir kesin kategori etiketi, "
            f"**{summary['no_confirmed_category_label']}** tanesinde bu altı kategoride kesin etiket yok. "
            f"**{summary['uncertain_image_count']} görüntüde belirsiz etiket** ayrı tutuldu; bunlar kesin kategori toplamlarına dahil edilmedi. "
            f"Toplam **{summary['confirmed_label_assignments']} kategori ataması** var. Bir görüntü birden fazla kategoriye girebildiği "
            'için kategori sayıları örtüşür; toplamları farklı hatalı görüntü sayısı değildir. '
            'İnsan kontrolüne işaretlenmeyen kayıtların ilk görsel etiketleri korundu; sonuçlar bu 40 görüntüyü betimler, tüm val kümesine genellenmez.')


def uncertain_table(records):
    lines = ['| Kayıt | Görüntü ID | Belirsiz etiket |', '|---|---|---|']
    lines += [f"| {row['number']:02d} | {row['image_id']} | "
              f"{', '.join('belirsiz — '+category for category in row.get('uncertain_labels', []))} |"
              for row in records if row.get('uncertain_labels')]
    return lines


def load_review():
    """annotations.json is canonical; stop before writing on conflicting human decisions.

    review40.json supplies immutable image IDs, captions and references. The initial
    records_pending.json is only a fallback for first-time creation, never required
    by the delivery package once review40.json exists.
    """
    folder = STUDY/'review'
    annotation = read_json(folder/'annotations.json')
    current = read_json(folder/'review40.json') if (folder/'review40.json').exists() else None
    records = copy.deepcopy(current['records'] if current else read_json(folder/'records_pending.json'))
    decisions = annotation['annotations']
    if len(records) != 40 or len(decisions) != 40:
        raise ValueError('Review must contain exactly 40 records and annotations')
    numbers = [row['number'] for row in records]
    if numbers != list(range(1, 41)) or len({row['image_id'] for row in records}) != 40:
        raise ValueError('Duplicate or unordered review numbers/image IDs')
    by_number = {row['number']: row for row in decisions}
    if len(by_number) != 40 or set(by_number) != set(numbers):
        raise ValueError('Annotation numbers do not match records')
    if [row['image_id'] for row in records] != read_json(folder/'selection.json')['image_ids']:
        raise ValueError('Review image IDs do not match selection.json')
    categories = list(annotation['categories'])
    differences = []
    for row in records:
        decision = by_number[row['number']]
        labels, uncertain = decision['labels'], decision.get('uncertain_labels', [])
        if (not isinstance(labels, list) or not isinstance(uncertain, list)
                or len(set(labels)) != len(labels) or len(set(uncertain)) != len(uncertain)
                or not set(labels).issubset(categories)
                or not set(uncertain).issubset(categories + EXTRA_UNCERTAIN_CATEGORIES)
                or set(labels) & set(uncertain)):
            raise ValueError(f"Record {row['number']:02d}: invalid or overlapping confirmed/uncertain labels")
        if len(row['references']) != 5:
            raise ValueError(f"Record {row['number']:02d}: five references required")
        if current and row.get('human_reviewed'):
            before, after = reviewed_decision(row), reviewed_decision(decision)
            for key in DECISION_FIELDS:
                if before[key] != after[key]:
                    differences.append(f"Record {row['number']:02d}, {key}: review40.json={before[key]!r}; annotations.json={after[key]!r}")
        row.update(decision)
    csv_path = folder/'review40.csv'
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            csv_rows = list(reader)
        if len(csv_rows) != 40 or [row['image_id'] for row in csv_rows] != [row['image_id'] for row in records]:
            raise ValueError('CSV image IDs do not match review records')
        for row, existing in zip(records, csv_rows):
            if not row.get('human_reviewed') and existing.get('review_status') not in ('reviewed', 'reviewed_uncertain'):
                continue
            expected = {category: str(int(category in row['labels'])) for category in categories}
            expected.update({key: str(row.get(key, '')) for key in
                             ('reason', 'human_labels', 'human_notes', 'uncertainty', 'review_status')})
            expected.update(uncertain_labels=', '.join(row.get('uncertain_labels', [])),
                            needs_user_review=str(row['needs_user_review']))
            for key, value in expected.items():
                if existing.get(key, '') != value:
                    differences.append(f"Record {row['number']:02d}, {key}: CSV={existing.get(key)!r}; annotations.json={value!r}")
    if differences:
        raise ValueError('Human decision conflict; no review output was written:\n' + '\n'.join(differences))
    return annotation, records


def finish_review():
    annotation, records = load_review()  # All conflict checks precede writes.
    folder = STUDY/'review'
    categories = list(annotation['categories'])
    fields = ['number','image_id','beam3',*categories,'reason','needs_user_review','uncertainty',
              'human_labels','human_notes',*[f'reference_{i}' for i in range(1,6)],
              'uncertain_labels','review_status']
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in records:
        record = {key: row.get(key, '') for key in
                  ('number','image_id','beam3','reason','needs_user_review','uncertainty','human_labels','human_notes','review_status')}
        record.update({category: int(category in row['labels']) for category in categories})
        record.update({f'reference_{i}': text for i,text in enumerate(row['references'],1)})
        record['uncertain_labels'] = ', '.join(row.get('uncertain_labels', []))
        writer.writerow(record)
    csv_bytes = output.getvalue().encode('utf-8-sig')
    # Count the actual CSV representation; uncertain labels live in a separate column.
    csv_rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    counts = {category: sum(int(row[category]) for row in csv_rows) for category in categories}
    previous = read_json(folder/'summary.json') if (folder/'summary.json').exists() else {}
    reviewed = [row['number'] for row in records if row.get('human_reviewed')]
    initial = previous.get('initial_pending_numbers',
                           [row['number'] for row in records if row.get('human_reviewed') or row['needs_user_review']])
    pending = [row['image_id'] for row in records if row['needs_user_review']]
    uncertain_rows = [row for row in records if row.get('uncertain_labels')]
    any_label = sum(any(int(row[category]) for category in categories) for row in csv_rows)
    summary = {**previous, 'n':len(records), 'counts':counts, 'multi_label':True,
               'any_label':any_label, 'no_confirmed_category_label':len(records)-any_label,
               'needs_user_review':pending, 'initial_pending_numbers':initial, 'reviewed_numbers':reviewed,
               'uncertain_counts':{c:sum(c in row.get('uncertain_labels', []) for row in records)
                                   for c in categories + EXTRA_UNCERTAIN_CATEGORIES},
               'uncertain_image_count':len(uncertain_rows),
               'uncertain_image_ids':[row['image_id'] for row in uncertain_rows],
               'confirmed_label_assignments':sum(counts.values()),
               'count_source':'review40.csv', 'count_source_sha256':hashlib.sha256(csv_bytes).hexdigest(),
               'counts_status':('final after human decisions; computed from review40.csv; uncertain labels excluded'
                                if not pending else 'provisional; human review incomplete; uncertain labels excluded')}
    lines = ['# Model 2 beam-3: 40 görüntülük görsel inceleme', '',
             'Bu bir val örneklemi; test görüntüsü kullanılmadı. Tüm görüntüler, tahminler ve beşer referans '
             'ilk görsel incelemede birlikte değerlendirildi. İşaretlenen kayıtlar kullanıcı kararlarıyla '
             'sonuçlandırılır; etiketler bağımsız ground truth değildir.', '',
             'Seçim: 1.000 val ID üzerinde `SHA256("model2-beam3-review-20261112-v1:" + image_id)` '
             'artan sırası, ilk 40. Kalite filtresi yok. Seçim manifesti `selection.json`.', '',
             'Fotoğrafları açmak için yerel Flickr8k kurulumu gerekir; görüntü kimlikleri '
             '`data/flickr8k/Images/` altında aranır. Veri seti teslim paketine dahil değildir.', '',
             '| Kategori | Kesin sayı / 40 | Örnek görüntü ID | Kısa gerekçe |', '|---|---:|---|---|']
    for category in categories:
        example = next((row for row in records if category in row['labels']), None)
        lines.append(f"| {category} | {counts[category]} | {example['image_id'] if example else '—'} | "
                     f"{example['reason'] if example else 'Bu kategoride kesin hata etiketi yok.'} |")
    lines += ['', '## Kodlama kuralları', '']
    lines += [f'- **{category}:** {definition}' for category,definition in annotation['categories'].items()]
    lines += ['', review_overview(summary), '', '### Belirsiz kayıtlar — kesin toplamlardan hariç', '',
              *uncertain_table(records), '',
              'Nihai karar kaynağı `annotations.json`; `review40.json` görüntü, tahmin ve referansları da saklar. '
              'CSV’de `human_labels`, `human_notes`, `uncertain_labels` ve `review_status` insan kararlarını korur.', '']
    for row in records:
        label_text = ', '.join(row['labels'] + ['belirsiz — '+c for c in row.get('uncertain_labels', [])])
        status = ('Tamamlandı; karar kaydedildi.' if row.get('human_reviewed') else
                  'EVET' if row['needs_user_review'] else 'İlk görsel incelemede sonuçlandırıldı.')
        lines += [f"## {row['number']:02d} — {row['image_id']}", '',
                  f"**Görüntü kimliği:** `{row['image_id']}`", '',
                  f"**Beam-3:** {row['beam3']}", '',
                  f"**Etiketler:** {label_text or 'Bu altı kategoride kesin etiket yok'}", '',
                  f"**Gerekçe:** {row['reason']}", '', f"**Kullanıcı kontrolü:** {status}", '']
        if row.get('uncertainty'):
            lines += [f"**Not / belirsizlik:** {row['uncertainty']}", '']
        lines += ['**Beş referans:**', ''] + [f'- {text}' for text in row['references']] + ['']
    outputs = {'review40.json':(json.dumps({'methodology':annotation,'records':records}, ensure_ascii=False, indent=2)+'\n').encode('utf-8'),
               'summary.json':(json.dumps(summary, ensure_ascii=False, indent=2)+'\n').encode('utf-8'),
               'review40.csv':csv_bytes, 'review40.md':('\n'.join(lines)+'\n').encode('utf-8')}
    for name, content in outputs.items():
        path = folder/name
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
    print('ANNOTATION_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)


def attention_examples():
    import torch
    from PIL import Image
    from torchvision import transforms
    from src.data.cache import FeatureStore
    from src.data.vocab import Vocabulary
    from src.train import build_model
    from src.decode import generate_caption
    from src.visualize_attention import visualize_attention
    torch.set_num_threads(4)
    records = json.loads((STUDY/'review/review40.json').read_text(encoding='utf-8'))['records']
    device = 'cuda'
    checkpoint = ROOT/'checkpoints/model2_attention_lstm_full_20261104_01/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    vocab = Vocabulary.load(ROOT/'src/data/artifacts/vocab.json')
    model = build_model(state['config'], vocab).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    cache = ROOT/'data/features/model2_val_20261112'
    store = FeatureStore(cache/'resnet50_layer4.npy',cache/'resnet50_layer4_index.json')
    cropper = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224)])
    output = STUDY/'attention'
    output.mkdir(exist_ok=False)
    manifest = []
    # Examples are illustrative after review; the 40-image sample itself is unfiltered.
    for number in [1,15,29,33]:
        row = records[number-1]
        image_id = row['image_id']
        features = store[image_id].unsqueeze(0).to(device)
        caption, weights = generate_caption(model, features, dict(enumerate(vocab.itos)),
            start_id=vocab.start_idx, end_id=vocab.end_idx, pad_id=vocab.pad_idx,
            method='beam',beam_size=3,return_attention=True)
        attention_source = 'beam3 decoded path'
        if caption != row['beam3']:
            # Show the archived caption's attention explicitly, without claiming it was regenerated.
            tokens = torch.tensor([[vocab.start_idx] + [vocab.stoi[word] for word in row['beam3'].split()]],device=device)
            with torch.no_grad():
                _, forced = model(features,tokens[:,:-1])
            weights = forced[0].cpu()
            attention_source = 'teacher-forced archived beam3 caption (local decoding differs)'
        words = row['beam3'].split()
        assert weights.shape == (len(words),49)
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        torch.testing.assert_close(weights.sum(-1),torch.ones(len(words)))
        with Image.open(ROOT/'data/flickr8k/Images'/image_id) as raw:
            crop = cropper(raw.convert('RGB'))
        target = output/f'{number:02d}_{Path(image_id).stem}_beam3.png'
        figure = visualize_attention(crop, words,weights,output_path=target)
        figure.clear()
        manifest.append({'number':number,'image_id':image_id,'archived_beam3':row['beam3'],
                         'local_beam3':caption,'matches_archived_caption':caption==row['beam3'],
                         'attention_source':attention_source,'attention_shape':list(weights.shape),
                         'row_sums_close_to_one':True,'image_crop':'Resize(short edge 256), CenterCrop(224), RGB before normalization',
                         'figure':target.relative_to(ROOT).as_posix(),'labels':row['labels']})
    write_json(output/'manifest.json',manifest)
    print('ATTENTION_OK',len(manifest),flush=True)


def compare_results():
    import csv
    historical = json.loads((ROOT/'results/model2_attention_lstm/val_metrics_beam3.json').read_text())
    historical_predictions = json.loads((ROOT/'results/model2_attention_lstm/val_captions_beam3.json').read_text())
    rows, captions = [], {}
    for tag in ['beam3_recheck','beam5']:
        beam = 3 if tag=='beam3_recheck' else 5
        path = STUDY/tag/f'val_metrics_beam{beam}.json'
        quality = json.loads(path.read_text())
        run = json.loads((STUDY/tag/'val_metrics.json').read_text())[f'beam{beam}']
        prediction = json.loads((STUDY/tag/f'val_captions_beam{beam}.json').read_text())
        assert quality['n_images']==run['images']==1000
        assert quality['reference_sha256']==historical['reference_sha256']
        assert quality['split_sha256']==historical['split_sha256']
        assert set(prediction)==set(historical_predictions)
        captions[tag] = prediction
        rows.append({'run':tag,'beam_size':beam,**quality['metrics'],
                     'unique_captions':run['unique_captions'],'ms_per_image':run['ms_per_image'],
                     'decode_seconds_approx':run['ms_per_image'], 'images':run['images']})
    # 1000 images * milliseconds/image / 1000 = seconds.
    oldrun = json.loads((ROOT/'results/model2_attention_lstm/val_metrics.json').read_text())['beam3']
    old = {'run':'historical_beam3','beam_size':3,**historical['metrics'],
           'unique_captions':oldrun['unique_captions'],'ms_per_image':oldrun['ms_per_image'],
           'decode_seconds_approx':oldrun['ms_per_image'],'images':oldrun['images']}
    changed = [image_id for image_id in historical_predictions
               if historical_predictions[image_id]!=captions['beam3_recheck'][image_id]]
    write_json(STUDY/'comparison.json',{
        'rows':[old,*rows],
        'delta_beam5_minus_local_beam3': {key:rows[1][key]-rows[0][key] for key in historical['metrics'] if key in rows[1]},
        'local_beam3_vs_historical_changed_captions':len(changed),'changed_caption_ids':changed,
        'historical_timing_is_cross_environment':True,
        'meteor_local_status':'unavailable: Java executable not installed; both new local runs omit METEOR; historical beam3 METEOR preserved',
        'timing_scope':'synchronized decoder only, excludes ResNet50 extraction/cache construction/model loading/metrics',
        'all_runs_exact_same_val_ids':True,'references_and_split_hashes_match':True,
    })
    fields = list(dict.fromkeys(key for row in [old,*rows] for key in row))
    with (STUDY/'comparison.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows([old,*rows])
    audit_data = json.loads((STUDY/'audit.json').read_text())
    assert all(digest(ROOT/path)==sha for path,sha in audit_data['nucleus_files_sha256'].items())
    print('COMPARISON_OK',json.dumps({'rows':[old,*rows],'historical_changed':len(changed)}),flush=True)


def report():
    # Regenerate documentation only; no image/model access or prediction commands.
    _, review_records = load_review()
    comparison = json.loads((STUDY/'comparison.json').read_text())
    audit_data = json.loads((STUDY/'audit.json').read_text())
    review = json.loads((STUDY/'review/summary.json').read_text(encoding='utf-8'))
    if review['count_source_sha256'] != digest(STUDY/'review/review40.csv'):
        raise ValueError('Review CSV changed; run finish-review before report')
    attention = json.loads((STUDY/'attention/manifest.json').read_text(encoding='utf-8'))
    old, b3, b5 = comparison['rows']
    delta = comparison['delta_beam5_minus_local_beam3']
    ratio = b5['ms_per_image']/b3['ms_per_image']
    lines = ['# B — Model 2 tamamlayıcı val deneyleri ve görsel hata incelemesi', '',
             'Tarih: 12 Kasım 2026. Dal: `b/model2-val-beam5-review-20261112`; temel `origin/main` '
             '`a625d644d4e970e5b02bfbc2606666afcf90023a`. Kaynak model/decoder/eğitim kodu değiştirilmedi. '
             'Yeniden eğitim veya test kümesi üzerinde tahmin/değerlendirme yapılmadı. Commit, push ve PR yapılmadı.', '',
             '## Beam-5 deneyi', '',
             'Aynı yerel epoch-10 checkpoint, aynı 1.000 val görüntüsü, görüntü başına aynı beş referans ve '
             'aynı `src.predict`/`src.evaluate` kullanıldı. Val-ID kapsamı, referans ve split SHA-256 eşitliği '
             '`audit.json`, `data_provenance.json` ve metrik JSON dosyalarıyla denetlendi. '
             f"Checkpoint SHA-256: `{audit_data['checkpoint_sha256']}`. Eski koşu checkpoint hash'i kaydetmemiş; "
             'tarihsel dosya kimliği yol/epoch/mimari/sözlük ve aşağıdaki yeniden üretim kontrolüne dayanır.', '',
             'Kaggle arşivi resmi uç noktadan hesap girişi olmadan indirildi; arşivden yalnız 1.000 val JPEG '
             've özgün captions.txt çıkarıldı. Train/test görüntüleri çıkarılmadı. '
             'Yeni özellik önbelleği yalnız val için `[1000,2048,7,7]` float16: ResNet50 IMAGENET1K_V1, '
             'Resize(256), CenterCrop(224), ortak ImageNet normalizasyonu; çıkarım CUDA AMP, batch 64. '
             'Önbellek hazırlığı 24,6 saniye; decoder zamanlarına dahil değil.', '',
             'Yeni beam-3 ve beam-5 aynı önbelleği kullanır: CUDA RTX 3050 Ti Laptop, torch 2.6.0+cu118 / '
             'torchvision 0.21.0+cu118, batch 1, decoder AMP kapalı, en fazla start dahil 34 token, '
             'OMP/MKL thread sayısı 4. Beam sıralaması toplam log olasılığıdır; uzunluk normalizasyonu yok. '
             'Tek değiştirilen çözümleme parametresi beam genişliğidir.', '',
             '| Metrik | Arşiv beam-3 | Yerel beam-3 kontrol | Yerel beam-5 | Δ beam-5 − yerel beam-3 |',
             '|---|---:|---:|---:|---:|']
    for metric in ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','ROUGE-L','CIDEr']:
        cells = [f'{row[metric]:.4f}' if metric in row else '—' for row in [old,b3,b5]]
        diff = f'{delta[metric]:+.4f}' if metric in delta else '—'
        lines.append(f"| {metric} | {' | '.join(cells)} | {diff} |")
    for key,label,fmt in [('unique_captions','Farklı tam cümle / 1000','d'),('ms_per_image','ms/görüntü (yalnız decoder)','.2f')]:
        cells = [format(row[key],fmt) for row in [old,b3,b5]]
        difference = b5[key]-b3[key]
        lines.append(f"| {label} | {' | '.join(cells)} | {difference:+.2f} |")
    lines += ['', 'Metrikler ham pycocoevalcap 1.2 ölçeği; yüzde değil. Özel tokenları kaldırma, küçük harfe '
              'çevirme, noktalama kaldırma ve boşluk birleştirme aynı. CIDEr belge frekansları tüm 1.000 val '
              'görüntüsü üzerinde hesaplandı; boş tahminler dışlanmadı.', '',
              f"Yerel beam-3 kontrolünde arşivdekinden farklı cümle sayısı: **{comparison['local_beam3_vs_historical_changed_captions']}/1000**. "
              'Varsa ID listesi `comparison.json` içinde. Farkın kesin nedeni izole edilmedi; '
              'yeniden oluşturulan önbellek ve yazılım ortamı olası etkenlerdir. '
              'Beam genişliği karşılaştırmasında aynı yerel önbelleği kullanan yeni beam-3 tabanı esas alındı.', '',
              f"Beam-5, yerel beam-3'e göre BLEU-4'te {delta['BLEU-4']:+.4f}, CIDEr'de {delta['CIDEr']:+.4f}, "
              f"ROUGE-L'de {delta['ROUGE-L']:+.4f} değişim verdi. Farklı tam cümle sayısı {b3['unique_captions']} → "
              f"{b5['unique_captions']}. Tek zamanlama koşusunda beam-5/beam-3 süre oranı {ratio:.2f}; "
              'daha geniş arama bütün kalite ölçülerini birlikte iyileştirmedi. Bu sonuç tek başına beam-5 ile '
              'ana beam-3 seçiminin değiştirilmesini desteklemiyor.', '',
              f"Yaklaşık toplam decoder süresi beam-3 {b3['decode_seconds_approx']:.2f} sn, beam-5 "
              f"{b5['decode_seconds_approx']:.2f} sn. Bunlar CUDA senkronizasyonlu tek koşu süreleridir; "
              'model yükleme, görüntü/önbellek hazırlığı ve metrik hesaplama hariçtir. '
              'Arşivdeki 182,65 ms/görüntü farklı ortamdan geldiği için yeni süreye doğrudan hızlanma iddiası yapılmaz.', '',
              '**METEOR:** Java yürütülebiliri ortamda yok. Yeni beam-5 ve yeni beam-3 kontrol sonuçlarında '
              'METEOR eksik; arşiv beam-3 METEOR=0,1861 korunmuştur. METEOR ölçülmeden bu ölçütte beam-5 '
              'iyileşti/kötüleşti denemez. Eski 8 Kasım durum raporunun METEOR hesaplanmadı notu güncel '
              '`results/val_table.md` ve arşiv metrik JSON dosyaları açısından eskimiştir.', '',
              '## Rapora eklenebilecek kısa Model 1 / Model 2 yorumu', '',
              '> Aynı çözümleme yöntemiyle Model 2, Model 1’den daha yüksek referans uyumu ve daha fazla farklı '
              'cümle üretmiştir. Greedy’de BLEU-4 0,1063’ten 0,1698’e, CIDEr 0,2306’dan 0,4409’a; beam-3’te '
              'BLEU-4 0,1389’dan 0,2030’a, CIDEr 0,2778’den 0,4697’ye yükselmiştir. METEOR da greedy’de '
              '0,1555 → 0,1867, beam-3’te 0,1590 → 0,1861 olmuştur. Farklı cümle sayıları greedy’de '
              '176 → 877, beam-3’te 164 → 813’tür. Buna karşılık parametre sayısı 5.653.004’ten '
              '11.159.565’e ve arşiv decoder süreleri greedy’de 15,05 → 63,20, beam-3’te 45,96 → '
              '182,65 ms/görüntüye çıkmıştır. Ortak eğitim ayarları eşleşse de model kapasitesi ve decoder '
              'yapısı da değiştiği için kazancın yalnız attention’dan kaynaklandığı iddia edilemez.', '',
              'Kaynak: `results/val_table.md` ana tablosu ve model1/model2 `val_metrics_{greedy,beam3}.json` '
              'dosyaları. Bu karşılaştırma Model 1 için yeni deney yürütmez.', '',
              '### Config kontrolü', '',
              '`configs/model1.yaml` / `configs/model2.yaml` karşılaştırmasında farklı alanlar yalnız '
              '`name`, `model.type`, `model.num_layers` (Model 1: 1) ve `model.attention_dim` '
              '(Model 2: 512). Embedding 256, hidden 512, dropout 0,5 aynıdır. Şunlar birebir eşleşir:', '',
              '- seed 42, ortak ResNet50 özellikleri, tam eğitim verisi (`train_fraction=1.0`);',
              '- Adam lr=0.001; batch 32; üst sınır 30 epoch; grad clip 5; teacher forcing 1; AMP açık;',
              '- padding yok sayılan çapraz entropi; val_loss erken durdurma, patience 5;',
              '- checkpoint seçimi val CIDEr; cider_every 2; değerlendirme batch 64.', '',
              'Ortak `base.yaml` aynı split, sözlük, seed ve ön işlemeyi tanımlar. Yerel checkpoint içindeki '
              'config de `model2.yaml` ile birebir eşleşir. Bunlar config eşitliğini doğrular; eğitimlerin her '
              'rastlantısal yürütme ayrıntısının aynı olduğunu kanıtlamaz. Gerçekte epoch sayıları farklıdır '
              '(Model 1: 12, Model 2: 11); seçilen checkpoint her ikisinde epoch 10’dur. '
              'Model 1 nn.LSTM, Model 2 attention + LSTMCell kullanır; parametre sayıları eşit değildir. '
              'Bu nedenle saf, parametre eşleştirilmiş bir attention ablation deneyi değildir.', '',
              '## Nucleus sınırı ve koruma', '',
              'Mevcut top-p=0.9, seed 42/43/44 sonuçları yeniden çalıştırılmadı ve değiştirilmedi. '
              'Decoder her görüntü için yerel CPU rastgele üretecini aynı koşu seed’ine sıfırlar; '
              'görüntü başına bağımsız seed akışları yoktur. Üç seed ortalaması ve ddof=1 standart sapması '
              'yalnız bu üç tekrarı özetler; güven aralığı, görüntü bazlı belirsizlik veya anlamsal '
              'çeşitlilik kanıtı değildir. `audit.json` içindeki nucleus dosya hash’leri deney sonunda '
              'yeniden karşılaştırıldı ve değişmedi.', '',
              '## 40 görüntülük görsel hata incelemesi', '',
              'Seçim 1.000 val ID üzerinde sabit tuzlu SHA-256 sıralamasının ilk 40 ID’sidir; kaliteye göre '
              'eleme yok. Tüm görüntüler, arşiv beam-3 tahminleri ve beşer referans ilk incelemede birlikte değerlendirildi. '
              'İşaretlenen kayıtların insan kararları annotations.json içinde korunur. '
              'Sayılar güncel review40.csv dosyasının altı kategori sütunundaki 1 değerlerinden hesaplanır; '
              'uncertain_labels alanındaki belirsiz etiketler kesin toplamlara katılmaz.', '',
              '| Kategori | Kesin görüntü sayısı / 40 |', '|---|---:|']
    lines += [f'| {category} | {count} |' for category,count in review['counts'].items()]
    lines += ['', review_overview(review), '',
              '**Kodlama kuralları:** Tekil bir aktörü anlatıp diğerini atlamak tek başına sayı hatası sayılmaz. '
              'Ana etkinliği doğru veren kısa açıklamalarda salt ayrıntı eksikliği hata sayılmaz. '
              'Belirsiz etiketler ayrı tutulur; kişi niteliği (boy/girl) notu altı kesin kategorinin dışındadır.', '',
              '### Belirsiz olarak bırakılan kayıtlar', '', *uncertain_table(review_records), '',
              '[Görüntü kimlikleri, tahminler, beş referans, son etiket ve gerekçe](../results/model2_val_study_20261112/review/review40.md) '
              '· [Sayım kaynağı CSV](../results/model2_val_study_20261112/review/review40.csv) '
              '· [Makine okunabilir özet](../results/model2_val_study_20261112/review/summary.json). '
              'CSV’de human_labels ve human_notes alanları insan kararlarını içerir. '
              'Fotoğrafları açmak için yerel Flickr8k kurulumu gerekir; veri seti teslim paketinde bulunmaz.', '',
              '## Attention örnekleri', '',
              '40 örnek sabit kaldı; aşağıdaki dört örnek hata türlerini ve bir doğru örneği göstermek '
              'için inceleme sonrasında seçildi, temsili istatistiksel örneklem değildir. Aynı checkpoint ve '
              'val önbelleğiyle beam-3 kelimelerine hizalı `[kelime_sayısı,49]` ağırlıklar çizildi. '
              'Her satır sonlu, negatif olmayan ve toplamı ≈1; görsel aynı Resize/CenterCrop RGB kırpımıdır. '
              'Attention doğruluk veya nedensel açıklama garantisi değildir.', '']
    for row in attention:
        labels = ', '.join(row['labels']) or 'ana içerik doğru'
        lines.append(f"- [{row['number']:02d} — {row['image_id']}](../{row['figure']}): {labels}. "
                     f"Arşiv tahminiyle eşleşme: {row['matches_archived_caption']}; attention kaynağı: {row['attention_source']}.")
    lines += ['', '## Dosyalar ve tekrar çalıştırma', '',
              '- `results/model2_val_study_20261112/beam5/`: yeni tahminler, üretim ölçümleri ve altı kalite metriği.',
              '- `results/model2_val_study_20261112/beam3_recheck/`: aynı ortam/önbellekle kontrol koşusu.',
              '- `comparison.json`, `comparison.csv`, `audit.json`, `data_provenance.json`: karşılaştırma ve izlenebilirlik.',
              '- `review/`: seçim, ham metinler, ön etiketler, bütünleşik JSON/CSV/Markdown inceleme tablosu.',
              '- `attention/`: dört PNG ve manifest; eski attention/nucleus dosyaları korunur.',
              '- `report/tools/model2_val_study.py`: val hazırlama, seçim, rapor ve attention üretme aracı.', '',
              'Veri, ZIP, özellik önbelleği ve checkpoint `data/` / `checkpoints/` içinde ve Git tarafından '
              'ignore edilir; stage edilmedi. Üretilen rapor/deney dosyaları untracked durumdadır. '
              'Mevcut sonuçlar ve `results/val_table.md` üzerine yazılmadı.', '',
              'PowerShell, repo kökünden; aşağıdaki iki komut gerçekten çalıştırıldı. Mevcut çıktı dosyaları '
              'varken tekrar çalıştırmak hata verir; yeni koşu için farklı `--output-dir` kullanın:', '',
              '```powershell', "$env:OMP_NUM_THREADS = '4'", "$env:MKL_NUM_THREADS = '4'", "$env:PYTHONUTF8 = '1'",
              '.\\.venv\\Scripts\\python.exe -m src.predict --checkpoint checkpoints/model2_attention_lstm_full_20261104_01/best.pt --split val --method beam --beam-size 5 --features-dir data/features/model2_val_20261112 --output-dir results/model2_val_study_20261112/beam5 --device cuda',
              '.\\.venv\\Scripts\\python.exe -m src.evaluate --predictions results/model2_val_study_20261112/beam5/val_captions_beam5.json --model-name model2_attention_lstm --split val --method beam --beam-size 5 --output results/model2_val_study_20261112/beam5/val_metrics_beam5.json',
              '```', '', 'Bu değerlendirme komutu Java olmadığı için `--meteor` kullanmadı. '
              'Deney tamamlandı; eğitim komutu veya `--allow-test` çalıştırılmadı.']
    target = ROOT/'report/model2_b_tamamlayici_20261112.md'
    content = '\n'.join(lines)+'\n'
    if not target.exists() or target.read_text(encoding='utf-8') != content:
        target.write_text(content, encoding='utf-8')
    print('REPORT_OK',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'acquire', 'review', 'audit', 'finish-review', 'attention', 'compare', 'report'])
    stage = parser.parse_args().stage
    {'select': selection, 'acquire': acquire, 'review': review_material,
     'audit': audit, 'finish-review': finish_review,
     'attention': attention_examples, 'compare': compare_results, 'report': report}[stage]()
