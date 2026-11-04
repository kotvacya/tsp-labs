-- 1. Таблица пользователей
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(150) NOT NULL,
    role VARCHAR(30) NOT NULL CHECK (role IN ('RESEARCHER', 'OPERATOR', 'ADMIN')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Таблица датасетов
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id),
    name VARCHAR(120) NOT NULL UNIQUE,
    description TEXT,
    storage_dir VARCHAR(255) NOT NULL,
    total_samples INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Таблица кадров датасета (фото + правильный ответ глубины)
CREATE TABLE IF NOT EXISTS dataset_samples (
    id SERIAL PRIMARY KEY,
    dataset_id INT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    sample_index INT NOT NULL,
    rgb_image_rel_path VARCHAR(255) NOT NULL,
    gt_depth_rel_path VARCHAR(255) NOT NULL,
    CONSTRAINT uq_dataset_sample UNIQUE (dataset_id, sample_index)
);

-- 4. Таблица моделей
CREATE TABLE IF NOT EXISTS neural_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    architecture VARCHAR(50) NOT NULL,
    backbone VARCHAR(50) NOT NULL,
    weights_rel_path VARCHAR(255) NOT NULL,
    hyperparameters JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Таблица экспериментов
CREATE TABLE IF NOT EXISTS benchmark_experiments (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id),
    dataset_id INT NOT NULL REFERENCES datasets(id),
    title VARCHAR(200) NOT NULL,
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    started_at TIMESTAMP,
    finished_at TIMESTAMP
);

-- 6. Связующая таблица Многие-ко-Многим (результат работы модели в эксперименте)
CREATE TABLE IF NOT EXISTS experiment_model_runs (
    id SERIAL PRIMARY KEY,
    experiment_id INT NOT NULL REFERENCES benchmark_experiments(id) ON DELETE CASCADE,
    model_id INT NOT NULL REFERENCES neural_models(id),
    mean_silog FLOAT,
    mean_absrel FLOAT,
    mean_rmse FLOAT,
    mean_delta_1 FLOAT,
    avg_inference_time_ms FLOAT,
    run_status VARCHAR(30) DEFAULT 'PENDING',
    CONSTRAINT uq_experiment_model UNIQUE (experiment_id, model_id)
);

-- 7. Таблица покадровых результатов для детального анализа
CREATE TABLE IF NOT EXISTS metric_records (
    id BIGSERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES experiment_model_runs(id) ON DELETE CASCADE,
    sample_id INT NOT NULL REFERENCES dataset_samples(id) ON DELETE CASCADE,
    silog FLOAT NOT NULL,
    abs_rel FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    delta_1 FLOAT NOT NULL,
    inference_time_ms FLOAT NOT NULL
);
