# run.py
"""
Point d'entrée principal pour l'application.
Lance l'application Streamlit et sert de point d'intégration
entre les modules preprocessing et uploads.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path


def check_dependencies():
    """Vérifier les dépendances nécessaires"""
    # Mapping des noms de packages pip vers les noms d'import Python
    package_map = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',     
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'imbalanced-learn': 'imblearn',  
        'scipy': 'scipy',
        'openpyxl': 'openpyxl',
        'pyarrow': 'pyarrow'
    }
    
    missing = []
    for package, import_name in package_map.items():
        try:
            __import__(import_name)
            print(f"✅ {package} imported successfully as {import_name}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} not found (tried importing as {import_name})")
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def launch_streamlit(app_path: str = "app.py", port: int = 8501):
    """Lancer l'application Streamlit"""
    if not Path(app_path).exists():
        print(f"❌ File not found: {app_path}")
        return False
    
    print(f"\n🚀 Launching Streamlit app: {app_path}")
    print(f"🌐 Opening at: http://localhost:{port}")
    print("-" * 60)
    
    # Ouvrir le navigateur
    webbrowser.open(f"http://localhost:{port}")
    
    # Lancer Streamlit
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_path,
        "--server.port",
        str(port),
        "--server.address",
        "localhost"
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n✅ Application stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        return False


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="AI Experimentation Platform - Data Preprocessing"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)"
    )
    parser.add_argument(
        "--app",
        type=str,
        default="app.py",
        help="Path to Streamlit app (default: app.py)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 AI EXPERIMENTATION PLATFORM")
    print("=" * 60)
    
    # Vérifier les dépendances
    print("\n🔍 Checking dependencies...")
    print("-" * 60)
    
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again.")
        return 1
    
    print("\n" + "=" * 60)
    print("✅ All dependencies installed successfully!")
    print("=" * 60)
    
    if args.check_only:
        return 0
    
    # Lancer l'application
    return 0 if launch_streamlit(args.app, args.port) else 1


if __name__ == "__main__":
    sys.exit(main())