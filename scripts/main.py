from scripts.fetch_data import fetch_sensor_data, save_to_csv

def main():
    print("📡 Mengambil data sensor dari Firebase...")
    df = fetch_sensor_data()
    
    if df.empty:
        print("⚠️ Data kosong, coba cek Firebase lagi.")
    else:
        save_to_csv(df)
        print("✅ Data berhasil disimpan ke CSV!")
        print(df.head())  # tampilkan 5 baris pertama

if __name__ == "__main__":
    main()
