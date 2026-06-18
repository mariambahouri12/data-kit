# fix_dataframe_or.py
from pathlib import Path

def fix_app():
    """Corriger l'utilisation de 'or' avec des DataFrames dans app.py"""
    app_path = Path("app.py")
    
    if not app_path.exists():
        print("❌ app.py not found")
        return
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Rechercher et remplacer
    old_line = 'df = st.session_state.processed_data or st.session_state.current_data'
    new_line = 'df = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.current_data'
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed DataFrame 'or' issue in app.py")
    else:
        # Essayer une autre variante
        old_line_v2 = 'df = st.session_state.processed_data or st.session_state.current_data'
        if old_line_v2 in content:
            content = content.replace(old_line_v2, new_line)
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed DataFrame 'or' issue in app.py")
        else:
            print("⚠️ Could not find the line to fix.")
            print("   Please manually fix it in app.py")

if __name__ == "__main__":
    fix_app()
    print("\nNow run: streamlit run app.py")