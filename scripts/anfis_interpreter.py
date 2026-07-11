import simpful as sf
sf.suppress_banner = True


class WeatherANFIS:
    def __init__(self):
        self.FS = sf.FuzzySystem(show_banner=False)

        # Temperature
        T_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0,0,20,25),
            term="Dingin"
        )

        T_mid = sf.FuzzySet(
            function=sf.Triangular_MF(23,29,34),
            term="Normal"
        )

        T_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(32,36,50,50),
            term="Panas"
        )
        self.FS.add_linguistic_variable(
            "Temperature",
            sf.LinguisticVariable([T_low, T_mid, T_high], universe_of_discourse=[0, 50])
        )

        # Humidity
        H_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0,0,35,55),
            term="Kering"
        )

        H_mid = sf.FuzzySet(
            function=sf.Triangular_MF(45,65,80),
            term="Sedang"
        )

        H_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(75,85,100,100),
            term="Lembab"
        )
        self.FS.add_linguistic_variable(
            "Humidity",
            sf.LinguisticVariable([H_low, H_mid, H_high], universe_of_discourse=[0, 100])
        )

        # Windspeed
        W_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0,0,3,7),
            term="Pelan"
        )

        W_mid = sf.FuzzySet(
            function=sf.Triangular_MF(5,12,18),
            term="Sedang"
        )

        W_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(15,20,30,30),
            term="Kencang"
        )
        self.FS.add_linguistic_variable(
            "Windspeed",
            sf.LinguisticVariable([W_low, W_mid, W_high], universe_of_discourse=[0, 30])
        )

        # Irradiance (lux)
        L_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0,0,120,280),
            term="Gelap"
        )

        L_mid = sf.FuzzySet(
            function=sf.Triangular_MF(250,500,750),
            term="Sedang"
        )

        L_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(700,850,1000,1000),
            term="Terang"
        )
        self.FS.add_linguistic_variable(
            "Irradiance",
            sf.LinguisticVariable(
                [L_low, L_mid, L_high],
                universe_of_discourse=[0,1000]
            )
        )

        # Pressure (hPa)
        P_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(950,970,990,1002),
            term="Rendah"
        )

        P_mid = sf.FuzzySet(
            function=sf.Triangular_MF(998,1013,1025),
            term="Normal"
        )

        P_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(1020,1030,1055,1060),
            term="Tinggi"
        )
        self.FS.add_linguistic_variable(
            "Pressure",
            sf.LinguisticVariable(
                [P_low, P_mid, P_high],
                universe_of_discourse=[900,1060]
            )
        )

        # Output Risk
        Risk_low = sf.FuzzySet(
            function=sf.Trapezoidal_MF(0,0,20,40),
            term="Rendah"
        )

        Risk_mid = sf.FuzzySet(
            function=sf.Triangular_MF(30,50,70),
            term="Sedang"
        )

        Risk_high = sf.FuzzySet(
            function=sf.Trapezoidal_MF(60,80,100,100),
            term="Tinggi"
        )
        self.FS.add_linguistic_variable(
            "Risk",
            sf.LinguisticVariable([Risk_low, Risk_mid, Risk_high], universe_of_discourse=[0, 100])
        )

        rules = [
            #tinggi
            "IF (Pressure IS Rendah) AND (Humidity IS Lembab) THEN (Risk IS Tinggi)",
            "IF (Pressure IS Rendah) AND (Irradiance IS Gelap) THEN (Risk IS Tinggi)",
            "IF (Pressure IS Rendah) AND (Windspeed IS Sedang) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Windspeed IS Kencang) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Humidity IS Lembab) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Humidity IS Lembab) AND (Irradiance IS Gelap) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Irradiance IS Gelap) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Temperature IS Panas) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",
            "IF (Humidity IS Lembab) AND (Windspeed IS Sedang) AND (Pressure IS Rendah) THEN (Risk IS Tinggi)",

            #sedang
            "IF (Pressure IS Normal) AND (Humidity IS Lembab) THEN (Risk IS Sedang)",
            "IF (Pressure IS Normal) AND (Irradiance IS Gelap) THEN (Risk IS Sedang)",
            "IF (Humidity IS Lembab) AND (Irradiance IS Sedang) THEN (Risk IS Sedang)",
            "IF (Temperature IS Panas) AND (Humidity IS Sedang) THEN (Risk IS Sedang)",
            "IF (Temperature IS Normal) AND (Humidity IS Lembab) THEN (Risk IS Sedang)",
            "IF (Windspeed IS Sedang) THEN (Risk IS Sedang)",
            "IF (Pressure IS Rendah) THEN (Risk IS Sedang)",

            #rendah
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

    def interpret_weather(self, prediction, risk_value):
        temp = prediction.get("temperature", 0.0)
        hum = prediction.get("humidity", 0.0)
        wind = prediction.get("windSpeed", 0.0)
        pressure = prediction.get("pressure", 0.0)
        light = prediction.get("irradiance", 0.0)

        #rendah
        if risk_value < 35:
            if pressure >= 1015 and wind <= 5 and light >= 700:
                return "🌞 Aman: Cuaca cerah dan stabil, sangat baik untuk aktivitas luar ruangan."
            elif pressure >= 1010 and wind <= 8:
                return "🌤️ Aman: Cuaca relatif stabil dengan sedikit potensi perubahan."
            else:
                return "☁️ Aman: Risiko cuaca rendah."

        #sedang
        elif risk_value < 60:
            if pressure < 1010 and hum >= 75:
                return "☁️ Waspada: Kelembapan tinggi dan tekanan udara mulai menurun."
            elif light < 400:
                return "🌥️ Waspada: Intensitas cahaya rendah, langit cenderung mendung."
            else:
                return "⚠️ Waspada: Terdapat potensi perubahan kondisi cuaca."

        #tinggi
        elif risk_value < 80:
            if wind >= 15:
                return "🌬️ Risiko Tinggi: Kecepatan angin cukup tinggi, berhati-hati terhadap aktivitas luar ruangan."
            elif pressure < 1005 and hum >= 80:
                return "🌧️ Risiko Tinggi: Tekanan udara rendah disertai kelembapan tinggi mengindikasikan cuaca tidak stabil."
            else:
                return "⚠️ Risiko Tinggi: Kondisi atmosfer mulai tidak stabil."

        #sangat tinggi
        else:
            if wind >= 20:
                return "⛈️ Bahaya: Angin sangat kencang, hindari aktivitas di luar ruangan."
            elif pressure < 995:
                return "⛈️ Bahaya: Tekanan udara sangat rendah, berpotensi terjadi cuaca ekstrem."
            elif hum >= 90 and light <= 200:
                return "🌩️ Bahaya: Kelembapan sangat tinggi dan intensitas cahaya sangat rendah menunjukkan kondisi cuaca yang sangat buruk."
            else:
                return "🚨 Bahaya: Kondisi atmosfer sangat tidak stabil, tingkat kewaspadaan tinggi."

    def evaluate(self, prediction):
        temperature = prediction.get("temperature", 0.0)
        humidity = prediction.get("humidity", 0.0)
        wind_speed = prediction.get("windSpeed", 0.0)
        irradiance = prediction.get("irradiance",0)
        pressure = prediction.get("pressure",0.0)

        # Supaya interpret_weather juga punya lightIntensity
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

        return {
            "risk": risk,
            "interpretation": interpretation,
            "Risk": risk,
            "Interpretasi": interpretation
        }