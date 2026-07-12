import simpful as sf
sf.suppress_banner = True


# ============================================================
# SUMBER KEBENARAN TUNGGAL UNTUK SEMUA AMBANG BATAS
# ============================================================
# Semua breakpoint fuzzy set DAN semua threshold di classify_condition()/
# interpret_weather() mengacu ke konstanta ini. Sebelumnya angka yang sama
# (mis. "tekanan rendah") didefinisikan dua kali dengan nilai berbeda di
# tempat berbeda -- itu bug laten yang membuat hasil interpretasi bisa
# tidak konsisten dengan kategori fuzzy-nya sendiri.
#
# CATATAN: nilai-nilai ini masih hasil estimasi manual (bukan hasil
# clustering dari data aktual stasiun cuaca). Untuk revisi berikutnya,
# nilai ini sebaiknya diganti dengan hasil analisis distribusi data
# historis Anda sendiri (lihat diskusi clustering yang sempat dibahas).

TEMP_DINGIN_AKHIR = 20
TEMP_NORMAL_TENGAH = 29
TEMP_PANAS_MULAI = 32

HUM_KERING_AKHIR = 35
HUM_SEDANG_TENGAH = 65
HUM_LEMBAB_MULAI = 75

WIND_PELAN_AKHIR = 3
WIND_SEDANG_TENGAH = 12
WIND_KENCANG_MULAI = 15

IRRADIANCE_GELAP_AKHIR = 280
IRRADIANCE_SEDANG_TENGAH = 500
IRRADIANCE_TERANG_MULAI = 700

PRESSURE_RENDAH_AKHIR = 1002
PRESSURE_NORMAL_TENGAH = 1013
PRESSURE_TINGGI_MULAI = 1020


class WeatherFuzzyInterpreter:
    """
    Sistem Inferensi Fuzzy MAMDANI (bukan ANFIS) untuk menerjemahkan
    hasil numerik model hybrid RNN-ANFIS menjadi skor risiko (0-100),
    narasi kondisi cuaca, dan kategori cuaca (Cerah/Berawan/Mendung/Hujan)
    yang bisa dipahami pengguna awam.

    PENTING (untuk penulisan Bab III/metodologi):
    Rule dan membership function di sini didesain manual berdasarkan
    pengetahuan domain (expert rules), BUKAN dipelajari/di-training dari
    data seperti model RNN-ANFIS untuk prediksi numerik. Ini adalah
    komponen terpisah dan sengaja tidak disebut "ANFIS" untuk menghindari
    kerancuan dengan model hybrid utama.

    KETERBATASAN: hardware stasiun cuaca (Tabel 1 pada proposal) tidak
    memiliki sensor curah hujan. Kategori "Hujan" pada classify_condition()
    adalah INFERENSI tidak langsung dari kombinasi kelembapan tinggi +
    tekanan rendah + cahaya rendah, bukan pengukuran hujan aktual.
    """

    def __init__(self):
        self.FS = sf.FuzzySystem(show_banner=False)

        # Temperature
        T_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0, 0, TEMP_DINGIN_AKHIR, TEMP_DINGIN_AKHIR + 5),
            term="Dingin"
        )
        T_mid = sf.FuzzySet(
            function=sf.Triangular_MF(TEMP_DINGIN_AKHIR + 3, TEMP_NORMAL_TENGAH, TEMP_PANAS_MULAI + 2),
            term="Normal"
        )
        T_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(TEMP_PANAS_MULAI, TEMP_PANAS_MULAI + 4, 50, 50),
            term="Panas"
        )
        self.FS.add_linguistic_variable(
            "Temperature",
            sf.LinguisticVariable([T_low, T_mid, T_high], universe_of_discourse=[0, 50])
        )

        # Humidity
        H_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0, 0, HUM_KERING_AKHIR, HUM_KERING_AKHIR + 20),
            term="Kering"
        )
        H_mid = sf.FuzzySet(
            function=sf.Triangular_MF(HUM_KERING_AKHIR + 10, HUM_SEDANG_TENGAH, HUM_LEMBAB_MULAI + 5),
            term="Sedang"
        )
        H_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(HUM_LEMBAB_MULAI, HUM_LEMBAB_MULAI + 10, 100, 100),
            term="Lembab"
        )
        self.FS.add_linguistic_variable(
            "Humidity",
            sf.LinguisticVariable([H_low, H_mid, H_high], universe_of_discourse=[0, 100])
        )

        # Windspeed
        W_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0, 0, WIND_PELAN_AKHIR, WIND_PELAN_AKHIR + 4),
            term="Pelan"
        )
        W_mid = sf.FuzzySet(
            function=sf.Triangular_MF(WIND_PELAN_AKHIR + 2, WIND_SEDANG_TENGAH, WIND_KENCANG_MULAI + 3),
            term="Sedang"
        )
        W_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(WIND_KENCANG_MULAI, WIND_KENCANG_MULAI + 5, 30, 30),
            term="Kencang"
        )
        self.FS.add_linguistic_variable(
            "Windspeed",
            sf.LinguisticVariable([W_low, W_mid, W_high], universe_of_discourse=[0, 30])
        )

        # Irradiance (lux)
        L_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0, 0, IRRADIANCE_GELAP_AKHIR - 160, IRRADIANCE_GELAP_AKHIR),
            term="Gelap"
        )
        L_mid = sf.FuzzySet(
            function=sf.Triangular_MF(IRRADIANCE_GELAP_AKHIR - 30, IRRADIANCE_SEDANG_TENGAH, IRRADIANCE_TERANG_MULAI + 50),
            term="Sedang"
        )
        L_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(IRRADIANCE_TERANG_MULAI, IRRADIANCE_TERANG_MULAI + 150, 1000, 1000),
            term="Terang"
        )
        self.FS.add_linguistic_variable(
            "Irradiance",
            sf.LinguisticVariable([L_low, L_mid, L_high], universe_of_discourse=[0, 1000])
        )

        # Pressure (hPa)
        P_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(950, 970, 990, PRESSURE_RENDAH_AKHIR),
            term="Rendah"
        )
        P_mid = sf.FuzzySet(
            function=sf.Triangular_MF(PRESSURE_RENDAH_AKHIR - 4, PRESSURE_NORMAL_TENGAH, PRESSURE_TINGGI_MULAI + 5),
            term="Normal"
        )
        P_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(PRESSURE_TINGGI_MULAI, PRESSURE_TINGGI_MULAI + 10, 1055, 1060),
            term="Tinggi"
        )
        self.FS.add_linguistic_variable(
            "Pressure",
            sf.LinguisticVariable([P_low, P_mid, P_high], universe_of_discourse=[900, 1060])
        )

        # Output Risk
        Risk_low = sf.FuzzySet(function=sf.Trapezoidal_MF(0, 0, 20, 40), term="Rendah")
        Risk_mid = sf.FuzzySet(function=sf.Triangular_MF(30, 50, 70), term="Sedang")
        Risk_high = sf.FuzzySet(function=sf.Trapezoidal_MF(60, 80, 100, 100), term="Tinggi")
        self.FS.add_linguistic_variable(
            "Risk",
            sf.LinguisticVariable([Risk_low, Risk_mid, Risk_high], universe_of_discourse=[0, 100])
        )

        rules = [
            # tinggi
            "IF (Pressure IS Rendah) AND (Humidity IS Lembab) THEN (Risk IS Tinggi)",
            "IF (Pressure IS Rendah) AND (Irradiance IS Gelap) THEN (Risk IS Tinggi)",
            "IF (Pressure IS Rendah) AND (Windspeed IS Sedang) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Windspeed IS Kencang) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Humidity IS Lembab) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Humidity IS Lembab) AND (Irradiance IS Gelap) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Irradiance IS Gelap) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Windspeed IS Sedang) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",

            # sedang
            "IF (Pressure IS Normal) AND (Humidity IS Lembab) THEN (Risk IS Sedang)",
            "IF (Pressure IS Normal) AND (Irradiance IS Gelap) THEN (Risk IS Sedang)",
            "IF (Humidity IS Lembab) AND (Irradiance IS Sedang) THEN (Risk IS Sedang)",
            "IF (Temperature IS Panas) AND (Humidity IS Sedang) THEN (Risk IS Sedang)",
            "IF (Temperature IS Normal) AND (Humidity IS Lembab) THEN (Risk IS Sedang)",
            "IF (Windspeed IS Sedang) THEN (Risk IS Sedang)",
            "IF (Pressure IS Rendah) THEN (Risk IS Sedang)",

            # rendah
            "IF (Pressure IS Tinggi) AND (Windspeed IS Pelan) THEN (Risk IS Rendah)",
            "IF (Pressure IS Normal) AND (Windspeed IS Pelan) THEN (Risk IS Rendah)",
            "IF (Pressure IS Tinggi) AND (Irradiance IS Terang) THEN (Risk IS Rendah)",
            "IF (Pressure IS Normal) AND (Irradiance IS Terang) THEN (Risk IS Rendah)",
            "IF (Temperature IS Normal) AND (Humidity IS Sedang) THEN (Risk IS Rendah)",
            "IF (Temperature IS Normal) AND (Pressure IS Normal) THEN (Risk IS Rendah)",
            "IF (Humidity IS Kering) AND (Irradiance IS Terang) THEN (Risk IS Rendah)",
            "IF (Humidity IS Sedang) AND (Windspeed IS Pelan) THEN (Risk IS Rendah)",
            "IF (Pressure IS Tinggi) AND (Humidity IS Sedang) THEN (Risk IS Rendah)",
            "IF (Temperature IS Normal) AND (Humidity IS Sedang) AND (Pressure IS Tinggi) THEN (Risk IS Rendah)",
            "IF (Temperature IS Normal) AND (Humidity IS Sedang) AND (Pressure IS Normal) THEN (Risk IS Rendah)",
            "IF (Temperature IS Normal) AND (Windspeed IS Pelan) THEN (Risk IS Rendah)",
            "IF (Pressure IS Tinggi) AND (Humidity IS Kering) THEN (Risk IS Rendah)",
            "IF (Pressure IS Normal) AND (Humidity IS Kering) THEN (Risk IS Rendah)",
        ]

        self.FS.add_rules(rules)

    # ------------------------------------------------------------
    # KATEGORI CUACA EKSPLISIT (untuk Tabel 16 proposal & dashboard)
    # ------------------------------------------------------------
    def classify_condition(self, prediction):
        """
        Mengembalikan salah satu dari: "Cerah", "Berawan", "Mendung", "Hujan".

        Dihitung dari threshold PLAIN (bukan lewat mesin fuzzy simpful),
        pakai konstanta yang SAMA dengan breakpoint fuzzy set di atas --
        supaya tidak ada dua definisi berbeda untuk batas yang sama.

        Lihat catatan keterbatasan di docstring kelas ini soal kategori
        "Hujan" yang sifatnya inferensi, bukan pengukuran langsung.
        """
        irradiance = prediction.get("irradiance", 0.0)
        humidity = prediction.get("humidity", 0.0)
        pressure = prediction.get("pressure", 0.0)

        if (
            humidity >= HUM_LEMBAB_MULAI
            and pressure <= PRESSURE_RENDAH_AKHIR
            and irradiance <= IRRADIANCE_GELAP_AKHIR
        ):
            return "Hujan"

        if irradiance <= IRRADIANCE_GELAP_AKHIR:
            return "Mendung"

        if irradiance <= IRRADIANCE_TERANG_MULAI:
            return "Berawan"

        return "Cerah"

    def interpret_weather(self, prediction, risk_value):
        temp = prediction.get("temperature", 0.0)
        hum = prediction.get("humidity", 0.0)
        wind = prediction.get("windSpeed", 0.0)
        pressure = prediction.get("pressure", 0.0)
        light = prediction.get("irradiance", 0.0)

        # rendah
        if risk_value < 35:
            if pressure >= PRESSURE_TINGGI_MULAI and wind <= WIND_PELAN_AKHIR + 2 and light >= IRRADIANCE_TERANG_MULAI:
                return "Aman: Cuaca cerah dan stabil, sangat baik untuk aktivitas luar ruangan."
            elif pressure >= PRESSURE_NORMAL_TENGAH - 3 and wind <= WIND_SEDANG_TENGAH - 4:
                return "Aman: Cuaca relatif stabil dengan sedikit potensi perubahan."
            else:
                return "Aman: Risiko cuaca rendah."

        # sedang
        elif risk_value < 60:
            if pressure < PRESSURE_NORMAL_TENGAH - 3 and hum >= HUM_LEMBAB_MULAI:
                return "Waspada: Kelembapan tinggi dan tekanan udara mulai menurun."
            elif light < IRRADIANCE_SEDANG_TENGAH - 100:
                return "Waspada: Intensitas cahaya rendah, langit cenderung mendung."
            else:
                return "Waspada: Terdapat potensi perubahan kondisi cuaca."

        # tinggi
        elif risk_value < 80:
            if wind >= WIND_KENCANG_MULAI:
                return "Risiko Tinggi: Kecepatan angin cukup tinggi, berhati-hati terhadap aktivitas luar ruangan."
            elif pressure < PRESSURE_RENDAH_AKHIR + 3 and hum >= HUM_LEMBAB_MULAI + 5:
                return "Risiko Tinggi: Tekanan udara rendah disertai kelembapan tinggi mengindikasikan cuaca tidak stabil."
            else:
                return "Risiko Tinggi: Kondisi atmosfer mulai tidak stabil."

        # sangat tinggi
        else:
            if wind >= WIND_KENCANG_MULAI + 5:
                return "Bahaya: Angin sangat kencang, hindari aktivitas di luar ruangan."
            elif pressure < PRESSURE_RENDAH_AKHIR - 7:
                return "Bahaya: Tekanan udara sangat rendah, berpotensi terjadi cuaca ekstrem."
            elif hum >= HUM_LEMBAB_MULAI + 15 and light <= IRRADIANCE_GELAP_AKHIR - 80:
                return "Bahaya: Kelembapan sangat tinggi dan intensitas cahaya sangat rendah menunjukkan kondisi cuaca yang sangat buruk."
            else:
                return "Bahaya: Kondisi atmosfer sangat tidak stabil, tingkat kewaspadaan tinggi."

    def evaluate(self, prediction):
        temperature = prediction.get("temperature", 0.0)
        humidity = prediction.get("humidity", 0.0)
        wind_speed = prediction.get("windSpeed", 0.0)
        irradiance = prediction.get("irradiance", 0.0)
        pressure = prediction.get("pressure", 0.0)

        prediction_safe = prediction.copy()
        prediction_safe["irradiance"] = irradiance

        self.FS.set_variable("Temperature", temperature)
        self.FS.set_variable("Humidity", humidity)
        self.FS.set_variable("Windspeed", wind_speed)
        self.FS.set_variable("Irradiance", irradiance)
        self.FS.set_variable("Pressure", pressure)

        risk = self.FS.Mamdani_inference(["Risk"])["Risk"]
        risk = round(float(risk), 2)

        interpretation = self.interpret_weather(prediction_safe, risk)
        kategori = self.classify_condition(prediction_safe)

        return {
            "risk": risk,
            "interpretation": interpretation,
            "kategori": kategori,
            # nama lama dipertahankan supaya field lama di Firebase/dashboard
            # yang sudah terlanjur dipakai tidak mendadak hilang
            "Risk": risk,
            "Interpretasi": interpretation,
        }