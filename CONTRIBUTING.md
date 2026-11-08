# Ekip içi çalışma

Bu dosya üç kişilik ekibin çalışma kurallarını toplar. Projenin kendisi, sabit kararlar ve tekrar çalıştırma adımları [README.md](README.md) içindedir.

## Dallar

- `main`: çalışan ortak kod
- `data`: Üye A
- `attention`: Üye B
- `transformer-demo`: Üye C

Çalışmaya başlamadan `main` dalını çekip kendi dalınıza alın. Bitince kendi dalınıza commit atın ve `main` dalına Pull Request açın.

## Dosya sahipliği

- `src/train.py` ve `src/data/`: Üye A
- `src/decode.py` ve attention modeli (`src/models/attention_lstm.py`): Üye B
- `src/evaluate.py` ve `app/`: Üye C

Başkasının dosyasında değişiklik gerekiyorsa dosyanın sahibi Pull Request'i birleştirir.

## Pull Request kuralı

- Her değişiklik kendi dalınızdan `main` dalına Pull Request ile gelir.
- Başkasının dosyasına dokunan Pull Request'i o dosyanın sahibi birleştirir.
- Genel özette "kilitli" yazan değerler ([README: Sabit kararlar](README.md#sabit-kararlar)) değiştirilmez.
- Veri, öznitelik önbelleği, checkpoint ve `kaggle.json` commit edilmez (`.gitignore`).
- `python -m src.data.prepare` çalıştırdıktan sonra `src/data/artifacts/` altında değişiklik görünürse commit etmeyin; `git checkout -- src/data/artifacts` ile geri alın.
