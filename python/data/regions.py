"""Region → LAD13NM council name mapping.

Mirrors ts/src/data/regions.ts — LAD13NM names match the martinjc UK-GeoJSON dataset.
"""

REGIONS: dict[str, list[str]] = {
    'London': [
        'Barnet', 'Camden', 'Hackney', 'Islington', 'Tower Hamlets',
        'Southwark', 'Lambeth', 'Westminster', 'Kensington and Chelsea',
        'Hammersmith and Fulham', 'Wandsworth', 'Lewisham', 'Greenwich',
        'Bexley', 'Bromley', 'Croydon', 'Merton', 'Kingston upon Thames',
        'Richmond upon Thames', 'Hounslow', 'Ealing', 'Hillingdon',
        'Harrow', 'Brent', 'Haringey', 'Enfield', 'Waltham Forest',
        'Redbridge', 'Havering', 'Barking and Dagenham', 'Newham',
        'City of London', 'Sutton',
    ],
    'South East': [
        'Brighton and Hove', 'Isle of Wight', 'Medway', 'Milton Keynes',
        'Oxford', 'Portsmouth', 'Reading', 'Slough', 'Southampton',
        'Bracknell Forest', 'West Berkshire', 'Windsor and Maidenhead', 'Wokingham',
        'Aylesbury Vale', 'Chiltern', 'South Bucks', 'Wycombe',
        'Eastbourne', 'Hastings', 'Lewes', 'Rother', 'Wealden',
        'Adur', 'Arun', 'Chichester', 'Crawley', 'Horsham', 'Mid Sussex', 'Worthing',
        'Ashford', 'Canterbury', 'Dartford', 'Dover', 'Gravesham', 'Maidstone',
        'Sevenoaks', 'Shepway', 'Swale', 'Thanet', 'Tonbridge and Malling', 'Tunbridge Wells',
        'Elmbridge', 'Epsom and Ewell', 'Guildford', 'Mole Valley', 'Reigate and Banstead',
        'Runnymede', 'Spelthorne', 'Surrey Heath', 'Tandridge', 'Waverley', 'Woking',
        'Basingstoke and Deane', 'East Hampshire', 'Eastleigh', 'Fareham', 'Gosport',
        'Hart', 'Havant', 'New Forest', 'Rushmoor', 'Test Valley', 'Winchester',
        'Cherwell', 'South Oxfordshire', 'Vale of White Horse', 'West Oxfordshire',
    ],
    'South West': [
        'Bath and North East Somerset', 'Bristol, City of', 'Cornwall', 'North Somerset',
        'Plymouth', 'South Gloucestershire', 'Swindon', 'Torbay', 'Wiltshire',
        'Bournemouth', 'Christchurch', 'Poole',
        'Dorset', 'Weymouth and Portland',
        'East Devon', 'Exeter', 'Mid Devon', 'North Devon', 'South Hams',
        'Teignbridge', 'Torridge', 'West Devon',
        'Cheltenham', 'Cotswold', 'Forest of Dean', 'Gloucester', 'Stroud', 'Tewkesbury',
        'Mendip', 'Sedgemoor', 'South Somerset', 'Taunton Deane', 'West Somerset',
        'Isles of Scilly',
    ],
    'East of England': [
        'Luton', 'Peterborough', 'Southend-on-Sea', 'Thurrock',
        'Bedford', 'Central Bedfordshire',
        'Cambridge', 'East Cambridgeshire', 'Fenland', 'Huntingdonshire', 'South Cambridgeshire',
        'Basildon', 'Braintree', 'Brentwood', 'Castle Point', 'Chelmsford', 'Colchester',
        'Epping Forest', 'Harlow', 'Maldon', 'Rochford', 'Tendring', 'Uttlesford',
        'Broxbourne', 'Dacorum', 'East Hertfordshire', 'Hertsmere', 'North Hertfordshire',
        'St Albans', 'Stevenage', 'Three Rivers', 'Watford', 'Welwyn Hatfield',
        'Broadland', 'Breckland', 'Great Yarmouth', "King's Lynn and West Norfolk",
        'North Norfolk', 'Norwich', 'South Norfolk',
        'Babergh', 'Forest Heath', 'Ipswich', 'Mid Suffolk', 'St Edmundsbury',
        'Suffolk Coastal', 'Waveney',
    ],
    'East Midlands': [
        'Derby', 'Leicester', 'Nottingham', 'Rutland',
        'Amber Valley', 'Bolsover', 'Chesterfield', 'Derbyshire Dales', 'Erewash',
        'High Peak', 'North East Derbyshire', 'South Derbyshire',
        'Blaby', 'Charnwood', 'Harborough', 'Hinckley and Bosworth', 'Melton',
        'North West Leicestershire', 'Oadby and Wigston',
        'Boston', 'East Lindsey', 'Lincoln', 'North Kesteven', 'South Holland',
        'South Kesteven', 'West Lindsey',
        'Corby', 'Daventry', 'East Northamptonshire', 'Kettering', 'Northampton',
        'South Northamptonshire', 'Wellingborough',
        'Ashfield', 'Bassetlaw', 'Broxtowe', 'Gedling', 'Mansfield',
        'Newark and Sherwood', 'Rushcliffe',
    ],
    'West Midlands': [
        'Birmingham', 'Coventry', 'Dudley', 'Sandwell', 'Solihull', 'Walsall', 'Wolverhampton',
        'Herefordshire, County of', 'Shropshire', 'Stoke-on-Trent', 'Telford and Wrekin',
        'Cannock Chase', 'East Staffordshire', 'Lichfield', 'Newcastle-under-Lyme',
        'South Staffordshire', 'Stafford', 'Staffordshire Moorlands', 'Tamworth',
        'North Warwickshire', 'Nuneaton and Bedworth', 'Rugby', 'Stratford-on-Avon', 'Warwick',
        'Bromsgrove', 'Malvern Hills', 'Redditch', 'Worcester', 'Wychavon', 'Wyre Forest',
    ],
    'Yorkshire & Humber': [
        'Barnsley', 'Bradford', 'Calderdale', 'Doncaster', 'Kirklees',
        'Leeds', 'Rotherham', 'Sheffield', 'Wakefield',
        'East Riding of Yorkshire', 'Kingston upon Hull, City of', 'York',
        'Craven', 'Hambleton', 'Harrogate', 'Richmondshire', 'Ryedale', 'Scarborough', 'Selby',
        'North East Lincolnshire', 'North Lincolnshire',
    ],
    'North West': [
        'Bolton', 'Bury', 'Manchester', 'Oldham', 'Rochdale', 'Salford',
        'Stockport', 'Tameside', 'Trafford', 'Wigan',
        'Knowsley', 'Liverpool', 'Sefton', 'St. Helens', 'Wirral',
        'Blackburn with Darwen', 'Blackpool', 'Cheshire East', 'Cheshire West and Chester',
        'Halton', 'Warrington',
        'Burnley', 'Chorley', 'Fylde', 'Hyndburn', 'Lancaster', 'Pendle', 'Preston',
        'Ribble Valley', 'Rossendale', 'South Ribble', 'West Lancashire', 'Wyre',
        'Allerdale', 'Barrow-in-Furness', 'Carlisle', 'Copeland', 'Eden', 'South Lakeland',
    ],
    'North East': [
        'Gateshead', 'Newcastle upon Tyne', 'North Tyneside', 'South Tyneside', 'Sunderland',
        'County Durham', 'Darlington', 'Hartlepool', 'Middlesbrough',
        'Northumberland', 'Redcar and Cleveland', 'Stockton-on-Tees',
    ],
}

MERGE_REGIONS: dict[str, list[str]] = {
    'Dorset': ['East Dorset', 'North Dorset', 'Purbeck', 'West Dorset'],
    'Chiltern and South Bucks': ['Chiltern', 'South Bucks'],
}

# Canonical region ordering (same as in the TS file)
REGION_ORDER: list[str] = list(REGIONS.keys())

# Inverse: LAD13NM → region name
COUNCIL_REGION: dict[str, str] = {
    lad_name: region
    for region, councils in REGIONS.items()
    for lad_name in councils
}
