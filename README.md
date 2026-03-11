<div align="center">

<a href="#english"><img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f1ec-1f1e7.svg" width="20" height="20" alt="EN"/> English</a>  |  <a href="#ukrainian"><img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f1fa-1f1e6.svg" width="20" height="20" alt="UA"/> Українська</a>  |  <a href="https://github.com/KirinDenis/ImageClusterizer/wiki">📖 Wiki</a>  |  <a href="https://github.com/KirinDenis/ImageClusterizer/releases">📥 Download Release</a>

</div>

---

<a id="english"></a>

# 🇬🇧 AI :: Image Clusterizer

> **AI-powered desktop application for organizing large image collections by visual similarity.**

## What Is This?

Image Clusterizer is a Windows desktop application that automatically groups your photos and images by visual content using deep learning. Instead of manually sorting thousands of images, you point the app at a folder and it does the work — scanning each image, extracting visual features with a neural network, and placing similar images close together on an interactive 2D map.

Whether you have a large photo archive, a dataset of product images, or thousands of screenshots to review — Image Clusterizer turns an overwhelming pile into a navigable visual landscape.

## Key Features

- **Visual similarity map** — images are projected onto a 2D canvas using PCA/RSVD; visually similar images appear near each other. Supports 200,000+ images with smooth zoom and pan.
- **Cluster view** — group images by similarity threshold into named clusters; browse cluster members as thumbnails.
- **AI-powered vectorization** — each image is processed by a pre-trained ResNet-50 CNN to extract a 2048-dimensional feature embedding or a 1000-dimensional logit vector.
- **GPU acceleration** — inference runs via ONNX Runtime with optional CUDA/DirectML GPU backend for fast batch processing.
- **Analysis Profiles** — save named presets (compression level, similarity threshold, vector type, GPU on/off) and switch between them with one click.
- **Live Telemetry cockpit** — real-time phase indicators (Scan / PCA / Render / Cluster), elapsed time, ETA, and throughput (images/sec).
- **Fast re-rendering** — PCA coordinates are cached in a local SQLite database; reloading an already-processed collection renders instantly without re-running the neural network.
- **Light / Dark theme** — switches at runtime with no restart required.

## How It Works

```

Folder of images
│
▼
ResNet-50 (ONNX)  ← GPU or CPU inference
│  2048-D embedding per image
▼
Sparse projection  ← top-N values kept (configurable, default 2048)
│
▼
RSVD (randomized SVD)  ← fast dimensionality reduction to 2D (Halko 2011)
│
▼
2D Map (MapRenderCanvas)  ← WPF ContainerVisual, hardware-accelerated
│
▼
Similarity clustering  ← cosine similarity, configurable threshold

```

## Technologies

| Area | Technology |
|------|-----------|
| Platform | Windows 10 / 11 |
| Framework | .NET 8.0 WPF |
| Architecture | MVVM (CommunityToolkit.Mvvm) |
| AI inference | ONNX Runtime 1.x |
| AI model | ResNet-50 v2 (ONNX Model Zoo) |
| GPU backend | CUDA / DirectML (optional) |
| Dimensionality reduction | Randomized SVD (MathNet.Numerics, Halko 2011) |
| Database | SQLite via Microsoft.Data.Sqlite |
| 2D rendering | WPF ContainerVisual + DrawingVisual |
| Thumbnails | System.Drawing (JPEG, configurable size) |
| Theming | WPF MergedDictionaries |

---

<a id="ukrainian"></a>

# 🇺🇦 AI :: Image Clusterizer

> **Десктопний застосунок на базі штучного інтелекту для організації великих колекцій зображень за візуальною схожістю.**

## Що це таке?

Image Clusterizer — це десктопний застосунок для Windows, який автоматично групує фотографії та зображення за їхнім візуальним змістом за допомогою алгоритмів глибокого навчання.

Замість ручного сортування тисяч файлів достатньо вказати папку із зображеннями — застосунок просканує її, проаналізує кожне зображення нейронною мережею, витягне візуальні ознаки та розташує схожі зображення поруч на інтерактивній двовимірній карті.

У результаті велика неструктурована колекція перетворюється на зручний візуальний простір для навігації та аналізу.

## Ключові можливості

- **Карта візуальної схожості** — зображення проєктуються на 2D-площину за допомогою PCA/RSVD. Підтримується понад 200 000 зображень із плавним масштабуванням і переміщенням.
- **Кластери** — автоматичне групування зображень за порогом схожості; перегляд мініатюр усередині кожного кластера.
- **AI-векторизація** — кожне зображення обробляється попередньо навченою моделлю ResNet-50, яка генерує 2048-вимірний ембедінг або 1000-вимірний логіт-вектор.
- **GPU-прискорення** — інференс виконується через ONNX Runtime із підтримкою CUDA або DirectML.
- **Профілі аналізу** — можливість зберігати та швидко перемикати набори параметрів аналізу.
- **Телеметрія в реальному часі** — відображення фаз обробки (Scan / PCA / Render / Cluster), часу виконання, ETA та швидкості обробки.
- **Швидке повторне відображення** — результати PCA кешуються у SQLite, тому повторне відкриття вже проаналізованої колекції відбувається миттєво.
- **Світла та темна тема** — перемикання інтерфейсу без перезапуску застосунку.

## Як це працює

```

Папка із зображеннями
│
▼
ResNet-50 (ONNX)  ← інференс на CPU або GPU
│
▼
2048-вимірний вектор ознак
│
▼
Sparse-проєкція  ← зберігаються лише Top-N значень
│
▼
RSVD (Randomized SVD)  ← зниження розмірності до 2D
│
▼
2D-карта (WPF ContainerVisual)
│
▼
Кластеризація за косинусною подібністю

```

## Технології

| Область | Технологія |
|--------|-----------|
| Платформа | Windows 10 / 11 |
| Фреймворк | .NET 8.0 WPF |
| Архітектура | MVVM (CommunityToolkit.Mvvm) |
| AI-інференс | ONNX Runtime |
| AI-модель | ResNet-50 v2 |
| GPU | CUDA / DirectML (опційно) |
| Зниження розмірності | Randomized SVD (MathNet.Numerics) |
| База даних | SQLite |
| Рендеринг | WPF ContainerVisual + DrawingVisual |
| Мініатюри | System.Drawing (JPEG) |
| Теми | WPF MergedDictionaries |

## Профілі аналізу

| Профіль | TopN | Поріг | Сценарій використання |
|--------|------|------|----------------|
| Default | 2048 | 0.85 | Загальне використання |
| Fast (compressed) | 256 | 0.80 | Швидкий попередній аналіз великих колекцій |
| Strict (deduplication) | 2048 | 0.97 | Пошук майже дублікатів зображень |
| Logit classes | 1000 | 0.75 | Групування за класами ImageNet |

## Зберігання даних

Усі дані зберігаються локально поруч із застосунком:

```

<папка застосунку>/
data/
vectors.db        ← SQLite: вектори, кеш PCA та метадані файлів
thumbnails/ <sha256>.jpg    ← JPEG-мініатюри
AppSettings.json    ← налаштування користувача та профілі

```

> Оригінальні файли зображень **ніколи не змінюються**.

## Polygon — дослідження та прототипи

Папка `Polygon/` містить окремі консольні C#-проєкти, які використовувалися для дослідження та перевірки алгоритмів перед інтеграцією в основний застосунок.

Детальніше — у Wiki.

## Ліцензія

MIT — див. файл LICENSE.
```


