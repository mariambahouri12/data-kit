# tracking/experiment_tracker.py
from typing import Dict, Any, Optional, List
import json
import sqlite3
import pandas as pd
from datetime import datetime
import os
import uuid

# Colonnes autorisées comme clés de filtre dans get_experiments(). Les noms de
# colonnes ne peuvent pas être paramétrés en SQL (seules les valeurs le
# peuvent) : on restreint donc `filters` à cette liste blanche pour éviter
# toute injection via une clé de dict non prévue.
_FILTERABLE_COLUMNS = {"run_id", "model_name", "task", "status"}


class ExperimentTracker:
    """
    Simple experiment tracking with SQLite.
    Tracks: runs, parameters, metrics, model versions.
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self._init_db()
        self.current_run_id = None
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT,
                task TEXT,
                status TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds REAL,
                metadata TEXT
            )
        """)
        
        # Parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                param_name TEXT,
                param_value TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Models table (versioning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                model_version TEXT,
                model_path TEXT,
                model_type TEXT,
                created_at TIMESTAMP,
                is_best BOOLEAN DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        # Tags table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                run_id TEXT,
                tag_key TEXT,
                tag_value TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (run_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start_run(self, model_name: str, task: str, params: Dict[str, Any], 
                  tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new experiment run"""
        run_id = str(uuid.uuid4())
        self.current_run_id = run_id
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert run
        cursor.execute("""
            INSERT INTO runs (run_id, model_name, task, status, start_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            model_name,
            task,
            'running',
            datetime.now().isoformat(),
            json.dumps({'tags': tags or {}})
        ))
        
        # Insert parameters
        for param_name, param_value in params.items():
            cursor.execute("""
                INSERT INTO parameters (run_id, param_name, param_value)
                VALUES (?, ?, ?)
            """, (run_id, param_name, json.dumps(param_value)))
        
        # Insert tags
        if tags:
            for tag_key, tag_value in tags.items():
                cursor.execute("""
                    INSERT INTO tags (run_id, tag_key, tag_value)
                    VALUES (?, ?, ?)
                """, (run_id, tag_key, tag_value))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float]):
        """Log metrics for a run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, metric_value in metrics.items():
            cursor.execute("""
                INSERT INTO metrics (run_id, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, (run_id, metric_name, metric_value))
        
        conn.commit()
        conn.close()
    
    def finish_run(self, run_id: str, status: str = 'completed'):
        """Mark run as completed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get start time
        cursor.execute("SELECT start_time FROM runs WHERE run_id = ?", (run_id,))
        start_time = cursor.fetchone()[0]
        
        # Calculate duration
        start = datetime.fromisoformat(start_time)
        end = datetime.now()
        duration = (end - start).total_seconds()
        
        # Update run
        cursor.execute("""
            UPDATE runs 
            SET status = ?, end_time = ?, duration_seconds = ?
            WHERE run_id = ?
        """, (status, end.isoformat(), duration, run_id))
        
        conn.commit()
        conn.close()
    
    def register_model(self, run_id: str, model_path: str, model_type: str, 
                       is_best: bool = False) -> str:
        """Register a trained model (versioning)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get next version
        cursor.execute("""
            SELECT COUNT(*) FROM models WHERE model_type = ?
        """, (model_type,))
        count = cursor.fetchone()[0]
        version = f"v{count + 1}.0"
        
        cursor.execute("""
            INSERT INTO models (run_id, model_version, model_path, model_type, created_at, is_best)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            version,
            model_path,
            model_type,
            datetime.now().isoformat(),
            1 if is_best else 0
        ))
        
        conn.commit()
        conn.close()
        
        return version
    
    def get_experiments(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get all experiments as DataFrame"""
        query = """
            SELECT 
                r.run_id,
                r.model_name,
                r.task,
                r.status,
                r.start_time,
                r.end_time,
                r.duration_seconds,
                GROUP_CONCAT(DISTINCT m.metric_name || ':' || m.metric_value) as metrics,
                GROUP_CONCAT(DISTINCT p.param_name || ':' || p.param_value) as parameters
            FROM runs r
            LEFT JOIN metrics m ON r.run_id = m.run_id
            LEFT JOIN parameters p ON r.run_id = p.run_id
        """
        
        query_params: List[Any] = []
        if filters:
            where_clauses = []
            for key, value in filters.items():
                if key not in _FILTERABLE_COLUMNS:
                    raise ValueError(
                        f"Filtre non autorisé: '{key}'. Colonnes valides: {sorted(_FILTERABLE_COLUMNS)}"
                    )
                where_clauses.append(f"r.{key} = ?")
                query_params.append(value)
            query += " WHERE " + " AND ".join(where_clauses)
        
        query += " GROUP BY r.run_id ORDER BY r.start_time DESC"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=query_params)
        conn.close()
        
        return df
    
    def get_best_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """Get the best model for a given type"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT run_id, model_version, model_path
            FROM models
            WHERE model_type = ? AND is_best = 1
            ORDER BY created_at DESC
            LIMIT 1
        """, (model_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'run_id': result[0],
                'version': result[1],
                'path': result[2]
            }
        return None
    
    def compare_experiments(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        conn = sqlite3.connect(self.db_path)
        
        # Get metrics for all runs
        placeholders = ','.join(['?'] * len(run_ids))
        query = f"""
            SELECT 
                run_id,
                metric_name,
                metric_value
            FROM metrics
            WHERE run_id IN ({placeholders})
        """
        
        df = pd.read_sql_query(query, conn, params=run_ids)
        
        # Pivot for comparison
        comparison = df.pivot(index='run_id', columns='metric_name', values='metric_value')
        
        conn.close()
        return comparison
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get all details for a specific run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get run info
        cursor.execute("""
            SELECT * FROM runs WHERE run_id = ?
        """, (run_id,))
        run = cursor.fetchone()
        
        if not run:
            return {}
        
        # Get parameters
        cursor.execute("""
            SELECT param_name, param_value FROM parameters WHERE run_id = ?
        """, (run_id,))
        params = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        
        # Get metrics
        cursor.execute("""
            SELECT metric_name, metric_value FROM metrics WHERE run_id = ?
        """, (run_id,))
        metrics = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get model info
        cursor.execute("""
            SELECT model_version, model_path, model_type FROM models WHERE run_id = ?
        """, (run_id,))
        model = cursor.fetchone()
        
        conn.close()
        
        return {
            'run_id': run[0],
            'model_name': run[1],
            'task': run[2],
            'status': run[3],
            'start_time': run[4],
            'end_time': run[5],
            'duration': run[6],
            'parameters': params,
            'metrics': metrics,
            'model': {
                'version': model[0] if model else None,
                'path': model[1] if model else None,
                'type': model[2] if model else None
            } if model else None
        }
