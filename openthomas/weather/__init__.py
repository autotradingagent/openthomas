"""Weather data layer: settlement stations, strike semantics, NWS ground
truth, Open-Meteo model guidance, and the prompt-facing WeatherDesk."""

from .baseline import strike_probability
from .desk import WeatherAssessment, WeatherDesk
from .nws import NWSClient
from .openmeteo import OpenMeteoClient
from .stations import STATIONS, Station, station_for_market, target_date, weather_series
from .strikes import Strike, parse_strike
from .verification import VerificationStore, prior_sigma

__all__ = [
    "STATIONS", "Station", "Strike", "NWSClient", "OpenMeteoClient",
    "VerificationStore", "WeatherAssessment", "WeatherDesk",
    "parse_strike", "prior_sigma", "station_for_market", "strike_probability",
    "target_date", "weather_series",
]
