import fastf1

fastf1.Cache.enable_cache("data/cache")

session = fastf1.get_session(2024, "Singapore", "R")
session.load()

print("Evento:", session.event["EventName"])
print("Pilotos:", list(session.results["Abbreviation"]))
print("Voltas carregadas:", len(session.laps))
print("Tem dados de clima:", not session.weather_data.empty)