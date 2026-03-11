<div align="center">

<a href="#english"><img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f1ec-1f1e7.svg" width="20" height="20" alt="EN"/> English</a>  |  <a href="#ukrainian"><img src="https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f1fa-1f1e6.svg" width="20" height="20" alt="UA"/> Українська</a>  |  <a href="https://github.com/KirinDenis/ImageClusterizer/wiki">📖 Wiki</a>  |  <a href="https://github.com/KirinDenis/ImageClusterizer/releases">📥 Download Release</a>

</div>

---

<a id="english"></a>

# AI :: Image Clusterizer

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

1. **Scan a folder** — the app walks the directory tree, runs ResNet-50 on each image, and stores the embedding vector and a thumbnail in a local SQLite database.
2. **PCA / RSVD** — if no cached 2D coordinates exist, randomized SVD reduces the high-dimensional vectors to 2D positions. Results are cached so the next load is instant.
3. **Explore the map** — zoom with the mouse wheel, pan by dragging, hover to preview, click to open in Explorer.
4. **Cluster** — switch to the Clusters tab, set a similarity threshold, click Compute.

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
| Theming | WPF MergedDictionaries, runtime-switchable |

## Getting Started

### Prerequisites
- Windows 10 or 11 (x64)
- [.NET 8 Desktop Runtime](https://dotnet.microsoft.com/download/dotnet/8.0)
- Optional: NVIDIA GPU with CUDA 11+ for hardware-accelerated inference

### 📥 Download
Go to [Releases](https://github.com/KirinDenis/ImageClusterizer/releases) and download the latest `ImageClusterizer.zip`. Extract and run `ImageClusterizer_WPF.exe`.

### Build from Source

```bash
git clone https://github.com/KirinDenis/ImageClusterizer.git
cd ImageClusterizer/ImageClusterizer

# Download the ONNX model (first time only):
curl -L -o ImageClusterizer_WPF/resnet50-v2-7.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"

dotnet run --project ImageClusterizer_WPF
```

### Quick Start

1. Launch the application.
2. Click **Scan folder** and select a folder containing images (JPG / PNG / BMP / WEBP).
3. Wait for vectorization to complete (progress bar + Live Telemetry shows speed and ETA).
4. The **Map** tab opens automatically with all images placed by visual similarity.
5. Use mouse wheel to zoom, drag to pan, hover for a preview, click to open the file in Explorer.
6. Switch to the **Clusters** tab, adjust the threshold slider, and click **Compute clusters**.
7. Click **Advanced** in the toolbar to open the cockpit: adjust compression, threshold, GPU, and manage Analysis Profiles.

## Analysis Profiles

The cockpit contains a **Profiles** panel. A profile stores:
- **SparseTopN** — how many vector dimensions are kept (lower = faster, less precise)
- **Similarity threshold** — cosine distance cutoff for grouping (0.50 – 0.99)
- **Vector type** — Embedding (2048-D) or Logit (1000-D)
- **Use GPU** flag

| Profile | TopN | Threshold | Use case |
|---------|------|-----------|----------|
| Default | 2048 | 0.85 | General purpose |
| Fast (compressed) | 256 | 0.80 | Quick preview of large datasets |
| Strict (deduplication) | 2048 | 0.97 | Finding near-duplicate images |
| Logit classes | 1000 | 0.75 | Grouping by ImageNet class label |

## Data Storage

All data is stored locally next to the executable:

```
<app folder>/
  data/
    vectors.db        ← SQLite: embeddings, PCA cache, file metadata
    thumbnails/
      <sha256>.jpg    ← JPEG thumbnails (generated once per image)
  AppSettings.json    ← user preferences and profiles
```

> Your original image files are **never modified**.

## Polygon — Research & Prototypes

The `Polygon/` folder contains a series of standalone C# console projects that were used to research and validate individual techniques before they were integrated into the main application.

| Project | What it demonstrates |
|---------|---------------------|
| [1. ResNet50_GetLogits_Test](Polygon/1.%20ResNet50_GetLogits_Test) | Running ResNet-50 inference and reading raw 1000-D logit output |
| [2. ResNet50_GetEmbedding_Test](Polygon/2.%20ResNet50_GetEmbedding_Test) | Extracting the 2048-D penultimate-layer embedding |
| [3. ResNet50_Image_similarity_search_test](Polygon/3.%20ResNet50_Image_similarity_search_test) | Cosine similarity search between image embeddings |
| [4. ResNet50_2D_PCA_test](Polygon/4.%20ResNet50_2D_PCA_test) | Reducing 2048-D vectors to 2D with PCA via SVD |
| [5. ResNet50_Sparse_Dot_Product_test](Polygon/5.%20ResNet50_Sparse_Dot_Product_test) | Sparse vector representation: keeping only top-N values |

## License

MIT — see [LICENSE](LICENSE).

---

<a id="ukrainian"></a>

# AI :: Image Clusterizer

> **Десктопний застосунок на базі штучного інтелекту для організації великих колекцій зображень за візуальною схожістю.**

## Що це таке?

Image Clusterizer — це Windows-застосунок, який автоматично групує фотографії та зображення за візуальним змістом за допомогою глибокого навчання. Замість ручного сортування тисяч зображень — ви просто вказуєте на папку, і застосунок робить все сам: сканує кожне зображення, витягує візуальні ознаки за допомогою нейромережі і розташовує схожі зображення поруч на інтерактивній 2D-карті.

Результат: з важкокерованого хаосу — зручний візуальний ландшафт, яким можна навігувати.

## Ключові можливості

- **Карта візуальної схожості** — зображення проєктуються на 2D-полотно через PCA/RSVD. Підтримка 200 000+ зображень з плавним зумом і перетягуванням.
- **Групи (кластери)** — групування зображень за порогом схожості; перегляд мініатюр членів кластера.
- **Векторизація на базі AI** — кожне зображення обробляється попередньо навченою моделлю ResNet-50, яка витягує 2048-D (ембедінг) або 1000-D (логіти) вектор.
- **GPU-прискорення** — інференс через ONNX Runtime з підтримкою CUDA/DirectML.
- **Профілі аналізу** — збережіть налаштування (стиснення, поріг, тип вектора, GPU) і перемикайтесь одним кліком.
- **Телеметрія в реальному часі** — індикатори фаз: Scan / PCA / Render / Cluster, час роботи, ETA, пропускна здатність.
- **Швидкий перерендеринг** — PCA-координати кешуються в SQLite; повторне завантаження вже обробленої папки — миттєве.
- **Світла / Темна тема** — перемикаються без перезапуску.

## Як це працює

```
Папка з зображеннями
       │
       ▼
ResNet-50 (ONNX)  ← інференс на GPU або CPU
       │  2048-D вектор на зображення
       ▼
Sparse проєкція  ← зберігаються Top-N значень
       │
       ▼
RSVD  ← швидке зниження розмірності до 2D (алгоритм Halko 2011)
       │
       ▼
2D Карта  ← WPF ContainerVisual, апаратне прискорення
       │
       ▼
Кластери  ← косинусна подібність, налаштовуваний поріг
```

## Технології

| Область | Технологія |
|--------|-----------|
| Платформа | Windows 10 / 11 |
| Фреймворк | .NET 8.0 WPF |
| Архітектура | MVVM (CommunityToolkit.Mvvm) |
| AI-інференс | ONNX Runtime 1.x |
| AI-модель | ResNet-50 v2 (ONNX Model Zoo) |
| GPU | CUDA / DirectML (необов'язково) |
| Зниження розмірності | Randomized SVD (MathNet.Numerics, Halko 2011) |
| База даних | SQLite (Microsoft.Data.Sqlite) |
| 2D-рендеринг | WPF ContainerVisual + DrawingVisual |
| Мініатюри | System.Drawing (JPEG) |
| Теми | WPF MergedDictionaries, перемикання без перезапуску |

## Початок роботи

### Вимоги
- Windows 10 або 11 (x64)
- [.NET 8 Desktop Runtime](https://dotnet.microsoft.com/download/dotnet/8.0)
- Опціонально: GPU NVIDIA з CUDA 11+ для апаратного інференсу

### 📥 Завантажити
Перейдіть у [розділ Releases](https://github.com/KirinDenis/ImageClusterizer/releases) і завантажте останній `ImageClusterizer.zip`. Розпакуйте і запустіть `ImageClusterizer_WPF.exe`.

### Збірка з вихідного коду

```bash
git clone https://github.com/KirinDenis/ImageClusterizer.git
cd ImageClusterizer/ImageClusterizer

# Завантажте ONNX-модель (лише першого разу):
curl -L -o ImageClusterizer_WPF/resnet50-v2-7.onnx \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"

dotnet run --project ImageClusterizer_WPF
```

### Швидкий старт

1. Запустіть застосунок.
2. Натисніть **Scan folder** і оберіть папку з зображеннями (JPG / PNG / BMP / WEBP).
3. Дочекайте завершення векторизації (прогрес-бар + Live Telemetry показують швидкість і ETA).
4. Вкладка **Map** відкривається автоматично — усі зображення розташовані за схожістю.
5. Колісо миші — зум, перетягування мишшю, наведення — прогляд, клік — відкриття в Explorer.
6. Вкладка **Clusters**: виставте поріг схожості, натисніть **Compute clusters**.
7. Кнопка **Advanced** — відкриває кокпіт: налаштування стиснення, порогу, GPU, Analysis Profiles.

## Профілі аналізу

Панель профілів зберігає:
- **SparseTopN** — кількість вимірів вектора, які зберігаються (менше = швидше, але менш точно)
- **Поріг схожості** — мінімальна косинусна відстань для потрапляння до кластера (0.50 – 0.99)
- **Тип вектора** — Embedding (2048-D) або Logit (1000-D)
- **Використовувати GPU**

| Профіль | TopN | Поріг | Сценарій використання |
|---------|------|-------|----------------|
| Default | 2048 | 0.85 | Загального призначення |
| Fast (compressed) | 256 | 0.80 | Швидкий перегляд великих колекцій |
| Strict (deduplication) | 2048 | 0.97 | Пошук понтових зображень |
| Logit classes | 1000 | 0.75 | Групування за категоріями ImageNet |

## Сховання даних

Усі дані зберігаються локально біля едзе:

```
<папка застосунку>/
  data/
    vectors.db        ← SQLite: вектори, кеш PCA, метадані файлів
    thumbnails/
      <sha256>.jpg    ← мініатюри JPEG
  AppSettings.json    ← налаштування та профілі
```

> Оригінальні файли зображень **ніколи не змінюються**.

## Polygon — Досліди і прототипи

Папка `Polygon/` містить серію самостійних C# консольних проектів — навчальні приклади окремих алгоритмів. Докладно висвітлені в [Wiki](https://github.com/KirinDenis/ImageClusterizer/wiki).

| Проект | Що демонструє |
|---------|---------------------|
| [1. ResNet50_GetLogits_Test](Polygon/1.%20ResNet50_GetLogits_Test) | Запуск ResNet-50 і отримання логітів 1000-D |
| [2. ResNet50_GetEmbedding_Test](Polygon/2.%20ResNet50_GetEmbedding_Test) | Витягування ембедінгу 2048-D |
| [3. ResNet50_Image_similarity_search_test](Polygon/3.%20ResNet50_Image_similarity_search_test) | Косинусна подібність між ембедінгами |
| [4. ResNet50_2D_PCA_test](Polygon/4.%20ResNet50_2D_PCA_test) | Зниження 2048-D до 2D через SVD |
| [5. ResNet50_Sparse_Dot_Product_test](Polygon/5.%20ResNet50_Sparse_Dot_Product_test) | Sparse вектори: збереження лише Top-N значень |

## Ліцензія

MIT — див. [LICENSE](LICENSE).
