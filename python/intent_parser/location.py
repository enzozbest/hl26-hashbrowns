"""UK location resolution — resolves fuzzy location text into structured council/area data.

Pure geography module with no external API dependencies.  Maps user location
text (council names, neighbourhood aliases, region names, city names) to
structured council lists and WGS84 coordinates.

Uses optional ``rapidfuzz`` for Levenshtein matching when available, falling
back to simple substring matching.
"""

from __future__ import annotations

import math
from typing import Optional

try:
    from rapidfuzz import fuzz as _fuzz

    def _fuzzy_score(a: str, b: str) -> float:
        """Return 0–1 similarity score using rapidfuzz."""
        return _fuzz.ratio(a, b) / 100.0

except ImportError:

    def _fuzzy_score(a: str, b: str) -> float:
        """Cheap fallback: normalised longest-common-substring ratio."""
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        if a in b or b in a:
            return min(len(a), len(b)) / max(len(a), len(b))
        # Character overlap ratio as last resort.
        common = sum(1 for c in set(a) if c in b)
        return common / max(len(set(a)), len(set(b)), 1)


# ---------------------------------------------------------------------------
# Council data — name → metadata
# ---------------------------------------------------------------------------

UK_COUNCILS: dict[str, dict] = {
    # ── London boroughs ────────────────────────────────────────────────
    "Camden": {
        "lat": 51.5517,
        "lng": -0.1588,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Camden Town", "Kentish Town", "Hampstead", "Gospel Oak",
            "King's Cross", "Bloomsbury", "Holborn", "Primrose Hill",
        ],
    },
    "Islington": {
        "lat": 51.5465,
        "lng": -0.1058,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Angel", "Highbury", "Holloway", "Finsbury Park",
            "Canonbury", "Barnsbury", "Clerkenwell",
        ],
    },
    "Hackney": {
        "lat": 51.5450,
        "lng": -0.0553,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Shoreditch", "Dalston", "Stoke Newington", "Clapton",
            "Homerton", "Hackney Wick", "London Fields", "Mare Street",
        ],
    },
    "Haringey": {
        "lat": 51.5906,
        "lng": -0.1110,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Tottenham", "Wood Green", "Muswell Hill", "Crouch End",
            "Hornsey", "Highgate", "Alexandra Park",
        ],
    },
    "Enfield": {
        "lat": 51.6538,
        "lng": -0.0799,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Edmonton", "Southgate", "Palmers Green", "Winchmore Hill",
            "Enfield Town", "Cockfosters",
        ],
    },
    "Barnet": {
        "lat": 51.6252,
        "lng": -0.1517,
        "region": "London",
        "sub_region": "North London",
        "country": "England",
        "aliases": [
            "Finchley", "Hendon", "Edgware", "High Barnet",
            "Golders Green", "Mill Hill", "Colindale", "Whetstone",
        ],
    },
    "Waltham Forest": {
        "lat": 51.5886,
        "lng": -0.0118,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Walthamstow", "Leyton", "Leytonstone", "Chingford",
            "Higham Hill", "William Morris",
        ],
    },
    "Tower Hamlets": {
        "lat": 51.5150,
        "lng": -0.0172,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Bethnal Green", "Bow", "Mile End", "Whitechapel",
            "Canary Wharf", "Poplar", "Stepney", "Limehouse", "Spitalfields",
        ],
    },
    "Newham": {
        "lat": 51.5255,
        "lng": 0.0352,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Stratford", "East Ham", "West Ham", "Forest Gate",
            "Plaistow", "Canning Town", "Beckton", "Royal Docks",
        ],
    },
    "Barking and Dagenham": {
        "lat": 51.5363,
        "lng": 0.0841,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Barking", "Dagenham", "Becontree", "Chadwell Heath",
            "Rush Green",
        ],
    },
    "Redbridge": {
        "lat": 51.5590,
        "lng": 0.0741,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Ilford", "Woodford", "Wanstead", "South Woodford",
            "Gants Hill", "Barkingside", "Hainault",
        ],
    },
    "Havering": {
        "lat": 51.5779,
        "lng": 0.2121,
        "region": "London",
        "sub_region": "East London",
        "country": "England",
        "aliases": [
            "Romford", "Hornchurch", "Upminster", "Harold Hill",
            "Rainham", "Collier Row",
        ],
    },
    "Westminster": {
        "lat": 51.4975,
        "lng": -0.1357,
        "region": "London",
        "sub_region": "Central London",
        "country": "England",
        "aliases": [
            "Mayfair", "Soho", "Covent Garden", "Marylebone",
            "Paddington", "Pimlico", "Victoria", "St James's",
            "Fitzrovia", "Bayswater",
        ],
    },
    "City of London": {
        "lat": 51.5155,
        "lng": -0.0922,
        "region": "London",
        "sub_region": "Central London",
        "country": "England",
        "aliases": [
            "The City", "Square Mile", "Bank", "Barbican",
            "Cheapside", "Moorgate", "Liverpool Street",
        ],
    },
    "Kensington and Chelsea": {
        "lat": 51.4990,
        "lng": -0.1938,
        "region": "London",
        "sub_region": "Central London",
        "country": "England",
        "aliases": [
            "Chelsea", "Kensington", "Notting Hill", "Earl's Court",
            "South Kensington", "Knightsbridge", "Holland Park",
            "North Kensington", "World's End",
        ],
    },
    "Lambeth": {
        "lat": 51.4571,
        "lng": -0.1231,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Brixton", "Clapham", "Streatham", "Vauxhall",
            "Waterloo", "Kennington", "Stockwell", "Herne Hill",
            "Tulse Hill", "Norwood",
        ],
    },
    "Southwark": {
        "lat": 51.4734,
        "lng": -0.0724,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Peckham", "Camberwell", "Bermondsey", "Dulwich",
            "Elephant and Castle", "Borough", "Rotherhithe",
            "Walworth", "Denmark Hill", "Nunhead",
        ],
    },
    "Lewisham": {
        "lat": 51.4415,
        "lng": -0.0117,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Deptford", "New Cross", "Catford", "Sydenham",
            "Brockley", "Forest Hill", "Blackheath", "Lee",
        ],
    },
    "Greenwich": {
        "lat": 51.4892,
        "lng": 0.0648,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Woolwich", "Eltham", "Charlton", "Plumstead",
            "Thamesmead", "Blackheath", "Greenwich Peninsula",
        ],
    },
    "Bromley": {
        "lat": 51.4039,
        "lng": 0.0198,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Beckenham", "Orpington", "Penge", "Crystal Palace",
            "Chislehurst", "Hayes", "Biggin Hill",
        ],
    },
    "Croydon": {
        "lat": 51.3714,
        "lng": -0.0977,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Thornton Heath", "Norbury", "Purley", "Coulsdon",
            "Addington", "South Norwood", "Selhurst",
        ],
    },
    "Sutton": {
        "lat": 51.3618,
        "lng": -0.1945,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Cheam", "Carshalton", "Wallington", "Belmont",
            "Beddington", "Worcester Park",
        ],
    },
    "Merton": {
        "lat": 51.4098,
        "lng": -0.1882,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Wimbledon", "Mitcham", "Morden", "Colliers Wood",
            "Raynes Park", "South Wimbledon",
        ],
    },
    "Kingston upon Thames": {
        "lat": 51.3925,
        "lng": -0.3057,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Kingston", "Surbiton", "New Malden", "Chessington",
            "Tolworth", "Norbiton",
        ],
    },
    "Wandsworth": {
        "lat": 51.4567,
        "lng": -0.1910,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Battersea", "Tooting", "Putney", "Balham",
            "Wandsworth Town", "Earlsfield", "Southfields",
            "Clapham Junction",
        ],
    },
    "Bexley": {
        "lat": 51.4549,
        "lng": 0.1505,
        "region": "London",
        "sub_region": "South London",
        "country": "England",
        "aliases": [
            "Bexleyheath", "Erith", "Sidcup", "Welling",
            "Crayford", "Thamesmead",
        ],
    },
    "Hammersmith and Fulham": {
        "lat": 51.4927,
        "lng": -0.2339,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Hammersmith", "Fulham", "Shepherd's Bush",
            "White City", "Brook Green", "Parsons Green",
            "Barons Court",
        ],
    },
    "Ealing": {
        "lat": 51.5130,
        "lng": -0.3089,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Acton", "Southall", "Hanwell", "Greenford",
            "Northolt", "Perivale",
        ],
    },
    "Hounslow": {
        "lat": 51.4668,
        "lng": -0.3612,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Chiswick", "Brentford", "Feltham", "Isleworth",
            "Heston", "Cranford", "Osterley",
        ],
    },
    "Hillingdon": {
        "lat": 51.5441,
        "lng": -0.4760,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Uxbridge", "Hayes", "Ruislip", "Heathrow",
            "Northwood", "Eastcote", "Ickenham",
        ],
    },
    "Brent": {
        "lat": 51.5588,
        "lng": -0.2817,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Wembley", "Kilburn", "Willesden", "Harlesden",
            "Neasden", "Kensal Green", "Stonebridge", "Kingsbury",
        ],
    },
    "Harrow": {
        "lat": 51.5898,
        "lng": -0.3346,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Harrow on the Hill", "Pinner", "Stanmore",
            "Wealdstone", "Kenton", "Edgware", "Hatch End",
        ],
    },
    "Richmond upon Thames": {
        "lat": 51.4479,
        "lng": -0.3260,
        "region": "London",
        "sub_region": "West London",
        "country": "England",
        "aliases": [
            "Richmond", "Twickenham", "Teddington", "Hampton",
            "Barnes", "Kew", "East Sheen", "Mortlake",
        ],
    },
    # ── Major cities / councils ────────────────────────────────────────
    "Manchester": {
        "lat": 53.4808,
        "lng": -2.2426,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": [
            "Manchester City Centre", "Ancoats", "Didsbury",
            "Chorlton", "Fallowfield", "Moss Side", "Hulme",
            "Northern Quarter", "Deansgate",
        ],
    },
    "Salford": {
        "lat": 53.4875,
        "lng": -2.2901,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": [
            "Media City", "Salford Quays", "Eccles", "Swinton",
            "Walkden", "Worsley", "Pendleton",
        ],
    },
    "Stockport": {
        "lat": 53.4106,
        "lng": -2.1575,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Cheadle", "Marple", "Bramhall", "Hazel Grove", "Reddish"],
    },
    "Tameside": {
        "lat": 53.4806,
        "lng": -2.0814,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Ashton-under-Lyne", "Hyde", "Stalybridge", "Denton", "Mossley"],
    },
    "Oldham": {
        "lat": 53.5409,
        "lng": -2.1114,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Chadderton", "Royton", "Shaw", "Failsworth", "Lees"],
    },
    "Rochdale": {
        "lat": 53.6097,
        "lng": -2.1561,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Heywood", "Middleton", "Littleborough", "Milnrow"],
    },
    "Bury": {
        "lat": 53.5933,
        "lng": -2.2966,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Ramsbottom", "Tottington", "Radcliffe", "Prestwich", "Whitefield"],
    },
    "Bolton": {
        "lat": 53.5785,
        "lng": -2.4299,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Farnworth", "Horwich", "Westhoughton", "Little Lever"],
    },
    "Wigan": {
        "lat": 53.5450,
        "lng": -2.6325,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": ["Leigh", "Hindley", "Standish", "Ashton-in-Makerfield", "Ince"],
    },
    "Trafford": {
        "lat": 53.4280,
        "lng": -2.3510,
        "region": "Greater Manchester",
        "sub_region": "Greater Manchester",
        "country": "England",
        "aliases": [
            "Stretford", "Sale", "Altrincham", "Urmston",
            "Old Trafford", "Timperley",
        ],
    },
    "Birmingham": {
        "lat": 52.4862,
        "lng": -1.8904,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": [
            "Edgbaston", "Erdington", "Moseley", "Selly Oak",
            "Handsworth", "Kings Heath", "Digbeth", "Jewellery Quarter",
            "Bournville",
        ],
    },
    "Wolverhampton": {
        "lat": 52.5870,
        "lng": -2.1288,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["Tettenhall", "Penn", "Bilston", "Wednesfield"],
    },
    "Coventry": {
        "lat": 52.4068,
        "lng": -1.5197,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["Earlsdon", "Coundon", "Stivichall", "Tile Hill"],
    },
    "Dudley": {
        "lat": 52.5119,
        "lng": -2.0810,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["Stourbridge", "Brierley Hill", "Halesowen", "Kingswinford"],
    },
    "Sandwell": {
        "lat": 52.5065,
        "lng": -1.9629,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["West Bromwich", "Smethwick", "Oldbury", "Rowley Regis", "Wednesbury"],
    },
    "Solihull": {
        "lat": 52.4130,
        "lng": -1.7780,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["Shirley", "Knowle", "Dorridge", "Solihull Town Centre"],
    },
    "Walsall": {
        "lat": 52.5860,
        "lng": -1.9822,
        "region": "West Midlands",
        "sub_region": "West Midlands",
        "country": "England",
        "aliases": ["Aldridge", "Bloxwich", "Brownhills", "Willenhall"],
    },
    "Leeds": {
        "lat": 53.8008,
        "lng": -1.5491,
        "region": "West Yorkshire",
        "sub_region": "West Yorkshire",
        "country": "England",
        "aliases": [
            "Headingley", "Chapel Allerton", "Roundhay", "Horsforth",
            "Morley", "Otley", "Kirkstall", "Hyde Park",
        ],
    },
    "Bradford": {
        "lat": 53.7960,
        "lng": -1.7594,
        "region": "West Yorkshire",
        "sub_region": "West Yorkshire",
        "country": "England",
        "aliases": ["Shipley", "Keighley", "Bingley", "Ilkley", "Manningham"],
    },
    "Wakefield": {
        "lat": 53.6830,
        "lng": -1.4956,
        "region": "West Yorkshire",
        "sub_region": "West Yorkshire",
        "country": "England",
        "aliases": ["Pontefract", "Castleford", "Normanton", "Ossett"],
    },
    "Kirklees": {
        "lat": 53.5933,
        "lng": -1.8013,
        "region": "West Yorkshire",
        "sub_region": "West Yorkshire",
        "country": "England",
        "aliases": ["Huddersfield", "Dewsbury", "Batley", "Cleckheaton", "Holmfirth"],
    },
    "Calderdale": {
        "lat": 53.7248,
        "lng": -1.8658,
        "region": "West Yorkshire",
        "sub_region": "West Yorkshire",
        "country": "England",
        "aliases": ["Halifax", "Hebden Bridge", "Todmorden", "Brighouse", "Elland"],
    },
    "Sheffield": {
        "lat": 53.3811,
        "lng": -1.4701,
        "region": "South Yorkshire",
        "sub_region": "South Yorkshire",
        "country": "England",
        "aliases": [
            "Hillsborough", "Ecclesall", "Crookes", "Broomhill",
            "Sharrow", "Kelham Island", "Meadowhall",
        ],
    },
    "Doncaster": {
        "lat": 53.5228,
        "lng": -1.1285,
        "region": "South Yorkshire",
        "sub_region": "South Yorkshire",
        "country": "England",
        "aliases": ["Balby", "Bessacarr", "Mexborough", "Conisbrough", "Thorne"],
    },
    "Rotherham": {
        "lat": 53.4326,
        "lng": -1.3635,
        "region": "South Yorkshire",
        "sub_region": "South Yorkshire",
        "country": "England",
        "aliases": ["Maltby", "Wath-upon-Dearne", "Rawmarsh", "Wickersley"],
    },
    "Barnsley": {
        "lat": 53.5529,
        "lng": -1.4793,
        "region": "South Yorkshire",
        "sub_region": "South Yorkshire",
        "country": "England",
        "aliases": ["Penistone", "Wombwell", "Hoyland", "Darfield"],
    },
    "Liverpool": {
        "lat": 53.4084,
        "lng": -2.9916,
        "region": "Merseyside",
        "sub_region": "Merseyside",
        "country": "England",
        "aliases": [
            "Anfield", "Toxteth", "Everton", "Wavertree",
            "Liverpool City Centre", "Baltic Triangle",
            "Woolton", "West Derby",
        ],
    },
    "Sefton": {
        "lat": 53.5034,
        "lng": -3.0084,
        "region": "Merseyside",
        "sub_region": "Merseyside",
        "country": "England",
        "aliases": ["Southport", "Bootle", "Crosby", "Formby", "Maghull"],
    },
    "Knowsley": {
        "lat": 53.4495,
        "lng": -2.8530,
        "region": "Merseyside",
        "sub_region": "Merseyside",
        "country": "England",
        "aliases": ["Kirkby", "Huyton", "Prescot", "Whiston", "Halewood"],
    },
    "St Helens": {
        "lat": 53.4534,
        "lng": -2.7368,
        "region": "Merseyside",
        "sub_region": "Merseyside",
        "country": "England",
        "aliases": ["Rainford", "Haydock", "Newton-le-Willows", "Eccleston"],
    },
    "Wirral": {
        "lat": 53.3727,
        "lng": -3.0738,
        "region": "Merseyside",
        "sub_region": "Merseyside",
        "country": "England",
        "aliases": ["Birkenhead", "Wallasey", "Heswall", "West Kirby", "Bebington"],
    },
    "Bristol": {
        "lat": 51.4545,
        "lng": -2.5879,
        "region": "South West",
        "sub_region": "Avon",
        "country": "England",
        "aliases": [
            "Clifton", "Redland", "Bedminster", "Stokes Croft",
            "Harbourside", "Bishopston", "Easton", "St Pauls",
        ],
    },
    "Newcastle upon Tyne": {
        "lat": 54.9783,
        "lng": -1.6178,
        "region": "North East",
        "sub_region": "Tyne and Wear",
        "country": "England",
        "aliases": [
            "Newcastle", "Jesmond", "Gosforth", "Heaton",
            "Byker", "Ouseburn", "Quayside", "Fenham",
        ],
    },
    "Nottingham": {
        "lat": 52.9548,
        "lng": -1.1581,
        "region": "East Midlands",
        "sub_region": "Nottinghamshire",
        "country": "England",
        "aliases": [
            "Lace Market", "Hockley", "Sneinton", "Beeston",
            "West Bridgford", "Sherwood", "Mapperley",
        ],
    },
    "Leicester": {
        "lat": 52.6369,
        "lng": -1.1398,
        "region": "East Midlands",
        "sub_region": "Leicestershire",
        "country": "England",
        "aliases": [
            "Stoneygate", "Clarendon Park", "Oadby",
            "Belgrave", "Highfields", "Knighton",
        ],
    },
    "Oxford": {
        "lat": 51.7520,
        "lng": -1.2577,
        "region": "South East",
        "sub_region": "Oxfordshire",
        "country": "England",
        "aliases": [
            "Jericho", "Cowley", "Headington", "Summertown",
            "Iffley", "Botley", "East Oxford",
        ],
    },
    "Cambridge": {
        "lat": 52.2053,
        "lng": 0.1218,
        "region": "East of England",
        "sub_region": "Cambridgeshire",
        "country": "England",
        "aliases": [
            "Trumpington", "Newnham", "Chesterton", "Cherry Hinton",
            "Mill Road", "Arbury", "Romsey",
        ],
    },
    "Brighton and Hove": {
        "lat": 50.8225,
        "lng": -0.1372,
        "region": "South East",
        "sub_region": "East Sussex",
        "country": "England",
        "aliases": [
            "Brighton", "Hove", "Kemptown", "Preston Park",
            "Rottingdean", "Hanover", "North Laine",
        ],
    },
    "Reading": {
        "lat": 51.4543,
        "lng": -0.9781,
        "region": "South East",
        "sub_region": "Berkshire",
        "country": "England",
        "aliases": [
            "Caversham", "Tilehurst", "Earley", "Woodley",
            "Whitley", "Southcote",
        ],
    },
    "Southampton": {
        "lat": 50.9097,
        "lng": -1.4044,
        "region": "South East",
        "sub_region": "Hampshire",
        "country": "England",
        "aliases": [
            "Shirley", "Portswood", "Woolston", "Bitterne",
            "Bassett", "Ocean Village", "Swaythling",
        ],
    },
    "Bath and North East Somerset": {
        "lat": 51.3758,
        "lng": -2.3599,
        "region": "South West",
        "sub_region": "Somerset",
        "country": "England",
        "aliases": [
            "Bath", "Keynsham", "Midsomer Norton", "Radstock",
            "Batheaston", "Paulton",
        ],
    },
    "York": {
        "lat": 53.9591,
        "lng": -1.0815,
        "region": "Yorkshire and the Humber",
        "sub_region": "North Yorkshire",
        "country": "England",
        "aliases": [
            "Clifton", "Bishopthorpe", "Acomb", "Heworth",
            "Tang Hall", "Fulford", "Heslington",
        ],
    },
    "Norwich": {
        "lat": 52.6309,
        "lng": 1.2974,
        "region": "East of England",
        "sub_region": "Norfolk",
        "country": "England",
        "aliases": [
            "Golden Triangle", "Eaton", "Thorpe Hamlet",
            "Lakenham", "Mousehold", "Bowthorpe",
        ],
    },
    "Exeter": {
        "lat": 50.7184,
        "lng": -3.5339,
        "region": "South West",
        "sub_region": "Devon",
        "country": "England",
        "aliases": [
            "St Thomas", "Heavitree", "Topsham", "Pinhoe",
            "Alphington", "Pennsylvania",
        ],
    },
    "Plymouth": {
        "lat": 50.3755,
        "lng": -4.1427,
        "region": "South West",
        "sub_region": "Devon",
        "country": "England",
        "aliases": [
            "Barbican", "Stonehouse", "Devonport", "Plympton",
            "Mutley", "Stoke", "Plymstock",
        ],
    },
    # ── Scotland ───────────────────────────────────────────────────────
    "Edinburgh": {
        "lat": 55.9533,
        "lng": -3.1883,
        "region": "Scotland",
        "sub_region": "Lothian",
        "country": "Scotland",
        "aliases": [
            "Leith", "Morningside", "Stockbridge", "New Town",
            "Old Town", "Bruntsfield", "Portobello", "Gorgie",
        ],
    },
    "Glasgow": {
        "lat": 55.8642,
        "lng": -4.2518,
        "region": "Scotland",
        "sub_region": "Central Scotland",
        "country": "Scotland",
        "aliases": [
            "West End", "Merchant City", "Finnieston", "Partick",
            "Govan", "Shawlands", "Hillhead", "Dennistoun",
        ],
    },
    # ── Wales ──────────────────────────────────────────────────────────
    "Cardiff": {
        "lat": 51.4816,
        "lng": -3.1791,
        "region": "Wales",
        "sub_region": "South Wales",
        "country": "Wales",
        "aliases": [
            "Cardiff Bay", "Canton", "Cathays", "Roath",
            "Splott", "Pontcanna", "Riverside", "Butetown",
        ],
    },
}


# ---------------------------------------------------------------------------
# Region → council mapping
# ---------------------------------------------------------------------------

REGIONS: dict[str, list[str]] = {
    # London sub-regions
    "South London": [
        "Lambeth", "Southwark", "Lewisham", "Greenwich", "Bromley",
        "Croydon", "Sutton", "Merton", "Kingston upon Thames",
        "Wandsworth", "Bexley",
    ],
    "North London": [
        "Camden", "Islington", "Haringey", "Enfield", "Barnet",
        "Hackney", "Waltham Forest",
    ],
    "East London": [
        "Tower Hamlets", "Newham", "Barking and Dagenham", "Redbridge",
        "Havering", "Hackney", "Waltham Forest",
    ],
    "West London": [
        "Hammersmith and Fulham", "Ealing", "Hounslow", "Hillingdon",
        "Brent", "Harrow", "Richmond upon Thames",
    ],
    "Central London": [
        "Westminster", "City of London", "Camden", "Islington",
        "Kensington and Chelsea", "Southwark", "Lambeth",
    ],
    "London": sorted({
        c for c, d in UK_COUNCILS.items() if d["region"] == "London"
    }),
    # Metropolitan areas
    "Greater Manchester": [
        "Manchester", "Salford", "Stockport", "Tameside", "Oldham",
        "Rochdale", "Bury", "Bolton", "Wigan", "Trafford",
    ],
    "West Midlands": [
        "Birmingham", "Wolverhampton", "Coventry", "Dudley",
        "Sandwell", "Solihull", "Walsall",
    ],
    "South Yorkshire": [
        "Sheffield", "Doncaster", "Rotherham", "Barnsley",
    ],
    "West Yorkshire": [
        "Leeds", "Bradford", "Wakefield", "Kirklees", "Calderdale",
    ],
    "Merseyside": [
        "Liverpool", "Sefton", "Knowsley", "St Helens", "Wirral",
    ],
}

# Auto-populate REGIONS from UK_COUNCILS for any region not already listed.
# This ensures every council's "region" value can be resolved as a region filter.
for _name, _data in UK_COUNCILS.items():
    _rgn = _data["region"]
    if _rgn not in REGIONS:
        REGIONS[_rgn] = []
    # Add council if not already present (avoids duplicates with hand-curated lists)
    if _name not in REGIONS[_rgn]:
        REGIONS[_rgn].append(_name)

    # Also index by sub_region if it differs from region and isn't already a key
    _sub = _data.get("sub_region")
    if _sub and _sub != _rgn and _sub not in REGIONS:
        REGIONS[_sub] = []
    if _sub and _sub != _rgn and _name not in REGIONS[_sub]:
        REGIONS[_sub].append(_name)


# ---------------------------------------------------------------------------
# Internal lookup indices (built once at import time)
# ---------------------------------------------------------------------------

# Lowercase council name → canonical council name
_COUNCIL_NAME_INDEX: dict[str, str] = {}

# Lowercase alias → canonical council name
_ALIAS_INDEX: dict[str, str] = {}

# Lowercase region name → canonical region name
_REGION_NAME_INDEX: dict[str, str] = {}

for _name, _data in UK_COUNCILS.items():
    _COUNCIL_NAME_INDEX[_name.lower()] = _name
    for _alias in _data.get("aliases", []):
        _lower = _alias.lower()
        # First council to claim an alias wins (some share, e.g. Blackheath).
        if _lower not in _ALIAS_INDEX:
            _ALIAS_INDEX[_lower] = _name

for _region in REGIONS:
    _REGION_NAME_INDEX[_region.lower()] = _region


# ---------------------------------------------------------------------------
# Radius suggestions by location level
# ---------------------------------------------------------------------------

_RADIUS_BY_LEVEL: dict[str, int] = {
    "address": 500,
    "neighbourhood": 1_000,
    "borough": 3_000,
    "city": 5_000,
    "region": 10_000,
    "county": 20_000,
    "country": 50_000,
    "unspecified": 5_000,
}


# ---------------------------------------------------------------------------
# Haversine helper
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    r = 6_371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve(raw_text: str) -> dict:
    """Resolve free-text location into structured council/area data.

    Tries, in order: exact council name, region name, neighbourhood alias,
    then fuzzy match against councils and aliases.

    Args:
        raw_text: Location text from the user, e.g. "South London",
                  "Shoreditch", "Hackney", "Greater Manchester".

    Returns:
        A dict with keys ``councils``, ``coordinates``, ``level``,
        ``radius_m``, and ``confidence``.
    """
    text = raw_text.strip()
    text_lower = text.lower()

    # 1. Exact council name match
    if text_lower in _COUNCIL_NAME_INDEX:
        name = _COUNCIL_NAME_INDEX[text_lower]
        data = UK_COUNCILS[name]
        return _make_result(
            councils=[name],
            lat=data["lat"],
            lng=data["lng"],
            level="borough" if data["region"] == "London" else "city",
            confidence=1.0,
        )

    # 2. Region match
    if text_lower in _REGION_NAME_INDEX:
        region = _REGION_NAME_INDEX[text_lower]
        councils = REGIONS[region]
        coords = councils_to_coordinates(councils)
        return _make_result(
            councils=councils,
            lat=coords["lat"],
            lng=coords["lng"],
            level="region",
            confidence=0.95,
        )

    # 3. Exact alias match
    if text_lower in _ALIAS_INDEX:
        name = _ALIAS_INDEX[text_lower]
        data = UK_COUNCILS[name]
        return _make_result(
            councils=[name],
            lat=data["lat"],
            lng=data["lng"],
            level="neighbourhood",
            confidence=0.90,
        )

    # 4. Substring match — check if query is contained in a council name
    #    or a council name is contained in the query.
    for council_lower, council_name in _COUNCIL_NAME_INDEX.items():
        if text_lower in council_lower or council_lower in text_lower:
            data = UK_COUNCILS[council_name]
            return _make_result(
                councils=[council_name],
                lat=data["lat"],
                lng=data["lng"],
                level="borough" if data["region"] == "London" else "city",
                confidence=0.80,
            )

    # 5. Substring match against aliases
    for alias_lower, council_name in _ALIAS_INDEX.items():
        if text_lower in alias_lower or alias_lower in text_lower:
            data = UK_COUNCILS[council_name]
            return _make_result(
                councils=[council_name],
                lat=data["lat"],
                lng=data["lng"],
                level="neighbourhood",
                confidence=0.75,
            )

    # 6. Fuzzy match against council names + aliases
    best_score = 0.0
    best_council: Optional[str] = None
    best_level = "borough"

    for council_lower, council_name in _COUNCIL_NAME_INDEX.items():
        score = _fuzzy_score(text_lower, council_lower)
        if score > best_score:
            best_score = score
            best_council = council_name
            data = UK_COUNCILS[council_name]
            best_level = "borough" if data["region"] == "London" else "city"

    for alias_lower, council_name in _ALIAS_INDEX.items():
        score = _fuzzy_score(text_lower, alias_lower)
        if score > best_score:
            best_score = score
            best_council = council_name
            best_level = "neighbourhood"

    for region_lower, region_name in _REGION_NAME_INDEX.items():
        score = _fuzzy_score(text_lower, region_lower)
        if score > best_score:
            best_score = score
            best_council = None
            best_level = "region"
            # Store region name for use below.
            _matched_region = region_name

    if best_score >= 0.6:
        if best_council is not None:
            data = UK_COUNCILS[best_council]
            return _make_result(
                councils=[best_council],
                lat=data["lat"],
                lng=data["lng"],
                level=best_level,
                confidence=round(best_score * 0.8, 2),  # discount for fuzzy
            )
        else:
            # Fuzzy-matched a region.
            councils = REGIONS[_matched_region]  # noqa: F821
            coords = councils_to_coordinates(councils)
            return _make_result(
                councils=councils,
                lat=coords["lat"],
                lng=coords["lng"],
                level="region",
                confidence=round(best_score * 0.8, 2),
            )

    # 7. Give up.
    return {
        "councils": [],
        "coordinates": None,
        "level": "unspecified",
        "radius_m": _RADIUS_BY_LEVEL["unspecified"],
        "confidence": 0.0,
    }


def get_nearby_councils(council_name: str, radius_km: int = 10) -> list[str]:
    """Return councils whose centre is within *radius_km* of the given council.

    Args:
        council_name: Canonical council name (must exist in ``UK_COUNCILS``).
        radius_km: Maximum distance in kilometres (default 10).

    Returns:
        List of nearby council names, sorted by distance (excludes the
        council itself).  Returns an empty list if *council_name* is unknown.
    """
    if council_name not in UK_COUNCILS:
        return []

    origin = UK_COUNCILS[council_name]
    lat1, lng1 = origin["lat"], origin["lng"]

    neighbours: list[tuple[float, str]] = []
    for name, data in UK_COUNCILS.items():
        if name == council_name:
            continue
        dist = _haversine_km(lat1, lng1, data["lat"], data["lng"])
        if dist <= radius_km:
            neighbours.append((dist, name))

    neighbours.sort()
    return [name for _, name in neighbours]


def councils_to_coordinates(council_names: list[str]) -> dict:
    """Return the centroid WGS84 lat/lng of the given councils.

    Unknown council names are silently skipped.

    Args:
        council_names: List of canonical council names.

    Returns:
        ``{"lat": float, "lng": float}`` — centroid of known councils,
        or ``{"lat": 0.0, "lng": 0.0}`` if none are known.
    """
    lats: list[float] = []
    lngs: list[float] = []
    for name in council_names:
        if name in UK_COUNCILS:
            lats.append(UK_COUNCILS[name]["lat"])
            lngs.append(UK_COUNCILS[name]["lng"])
    if not lats:
        return {"lat": 0.0, "lng": 0.0}
    return {
        "lat": round(sum(lats) / len(lats), 4),
        "lng": round(sum(lngs) / len(lngs), 4),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_result(
    councils: list[str],
    lat: float,
    lng: float,
    level: str,
    confidence: float,
) -> dict:
    """Build a standard result dict."""
    return {
        "councils": councils,
        "coordinates": {"lat": lat, "lng": lng},
        "level": level,
        "radius_m": _RADIUS_BY_LEVEL.get(level, 5_000),
        "confidence": confidence,
    }
