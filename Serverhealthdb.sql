-- ==========================================================
-- 🧠 CPUMETRIC DATABASE INITIALIZATION SCRIPT
-- ==========================================================

-- 1️⃣ Create the database
CREATE DATABASE IF NOT EXISTS CPUMETRIC;
USE CPUMETRIC;

-- ==========================================================
-- 2️⃣ Admin Table — Stores user credentials
-- ==========================================================
CREATE TABLE admin (
    admin_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE NOT NULL,
    phone VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 3️⃣ System Info — Each system registered under an admin
-- ==========================================================
CREATE TABLE system_info (
    system_id INT AUTO_INCREMENT PRIMARY KEY,
    system_name VARCHAR(100) NOT NULL,
    location VARCHAR(150),
    ip_address VARCHAR(50),
    admin_id INT NOT NULL,
    registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin(admin_id) ON DELETE CASCADE
);

-- ==========================================================
-- 4️⃣ System Metrics — Stores real-time CPU, Memory, etc.
-- ==========================================================
CREATE TABLE system_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    system_id INT NOT NULL,
    CPU_Usage FLOAT,
    Memory_Usage FLOAT,
    Disk_IO FLOAT,
    Network_Latency FLOAT,
    Error_Rate FLOAT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (system_id) REFERENCES system_info(system_id) ON DELETE CASCADE
);

-- ==========================================================
-- 5️⃣ Prediction Log — ML model predictions for each system
-- ==========================================================
CREATE TABLE prediction_log (
    prediction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    system_id INT NOT NULL,
    downtime_risk BOOLEAN NOT NULL DEFAULT 0,
    probability FLOAT,
    estimated_time_to_downtime INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (system_id) REFERENCES system_info(system_id) ON DELETE CASCADE
);

-- ==========================================================
-- 6️⃣ System History — Stores summarized performance timeline
-- ==========================================================
CREATE TABLE system_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    system_name VARCHAR(100),
    cpu_usage FLOAT,
    memory_usage FLOAT,
    disk_io FLOAT,
    network_latency FLOAT,
    error_rate FLOAT,
    status VARCHAR(20),
    downtime_detected BOOLEAN DEFAULT FALSE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================================
-- 7️⃣ Notifications — Generated alerts for high-risk systems
-- ==========================================================
CREATE TABLE notifications (
    notification_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    admin_id INT NOT NULL,
    system_id INT NOT NULL,
    message TEXT NOT NULL,
    risk_level ENUM('Low', 'Medium', 'High') DEFAULT 'Low',
    status ENUM('Unread', 'Read') DEFAULT 'Unread',
    sent_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (admin_id) REFERENCES admin(admin_id) ON DELETE CASCADE,
    FOREIGN KEY (system_id) REFERENCES system_info(system_id) ON DELETE CASCADE
);


