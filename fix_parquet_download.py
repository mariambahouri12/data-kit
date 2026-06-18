# fix_parquet_download.py
from pathlib import Path

def fix_app():
    """Corriger le téléchargement Parquet dans app.py"""
    app_path = Path("app.py")
    
    if not app_path.exists():
        print("❌ app.py not found")
        return
    
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Rechercher et remplacer la section Parquet
    old_parquet = '''with col2:
        parquet = df.to_parquet(index=False).getvalue()
        st.download_button(
            label="📥 Télécharger Parquet",
            data=parquet,
            file_name="processed_data.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )'''
    
    new_parquet = '''with col2:
        import io
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_data = parquet_buffer.getvalue()
        st.download_button(
            label="📥 Télécharger Parquet",
            data=parquet_data,
            file_name="processed_data.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )'''
    
    if old_parquet in content:
        content = content.replace(old_parquet, new_parquet)
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ Fixed Parquet download in app.py")
    else:
        # Essayer une autre variante
        old_parquet_v2 = '''with col2:
        parquet = df.to_parquet(index=False).getvalue()
        st.download_button(
            label="📥 Télécharger Parquet",
            data=parquet,
            file_name="processed_data.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )'''
        
        if old_parquet_v2 in content:
            content = content.replace(old_parquet_v2, new_parquet)
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✅ Fixed Parquet download in app.py")
        else:
            print("⚠️ Could not find Parquet download section to fix.")
            print("   Please manually fix it in app.py")

if __name__ == "__main__":
    fix_app()
    print("\nNow run: streamlit run app.py")