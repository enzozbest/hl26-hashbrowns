/** Maps each English region name to its constituent LAD names.
 *  Names match LAD13NM values from the martinjc UK-GeoJSON dataset. */
export const REGIONS: Record<string, string[]> = {
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
        // Unitaries
        'Brighton and Hove', 'Isle of Wight', 'Medway', 'Milton Keynes',
        'Oxford', 'Portsmouth', 'Reading', 'Slough', 'Southampton',
        // Berkshire
        'Bracknell Forest', 'West Berkshire', 'Windsor and Maidenhead', 'Wokingham',
        // Buckinghamshire
        'Aylesbury Vale', 'Chiltern', 'South Bucks', 'Wycombe',
        // East Sussex
        'Eastbourne', 'Hastings', 'Lewes', 'Rother', 'Wealden',
        // West Sussex
        'Adur', 'Arun', 'Chichester', 'Crawley', 'Horsham', 'Mid Sussex', 'Worthing',
        // Kent
        'Ashford', 'Canterbury', 'Dartford', 'Dover', 'Gravesham', 'Maidstone',
        'Sevenoaks', 'Shepway', 'Swale', 'Thanet', 'Tonbridge and Malling', 'Tunbridge Wells',
        // Surrey
        'Elmbridge', 'Epsom and Ewell', 'Guildford', 'Mole Valley', 'Reigate and Banstead',
        'Runnymede', 'Spelthorne', 'Surrey Heath', 'Tandridge', 'Waverley', 'Woking',
        // Hampshire
        'Basingstoke and Deane', 'East Hampshire', 'Eastleigh', 'Fareham', 'Gosport',
        'Hart', 'Havant', 'New Forest', 'Rushmoor', 'Test Valley', 'Winchester',
        // Oxfordshire
        'Cherwell', 'South Oxfordshire', 'Vale of White Horse', 'West Oxfordshire',
    ],
    'South West': [
        // Unitaries
        'Bath and North East Somerset', 'Bristol, City of', 'Cornwall', 'North Somerset',
        'Plymouth', 'South Gloucestershire', 'Swindon', 'Torbay', 'Wiltshire',
        // Dorset — 2013 boundaries (three separate councils)
        'Bournemouth', 'Christchurch', 'Poole',
        'East Dorset', 'North Dorset', 'Purbeck', 'West Dorset', 'Weymouth and Portland',
        // Devon
        'East Devon', 'Exeter', 'Mid Devon', 'North Devon', 'South Hams',
        'Teignbridge', 'Torridge', 'West Devon',
        // Gloucestershire
        'Cheltenham', 'Cotswold', 'Forest of Dean', 'Gloucester', 'Stroud', 'Tewkesbury',
        // Somerset
        'Mendip', 'Sedgemoor', 'South Somerset', 'Taunton Deane', 'West Somerset',
        // Other
        'Isles of Scilly',
    ],
    'East of England': [
        // Unitaries
        'Luton', 'Peterborough', 'Southend-on-Sea', 'Thurrock',
        // Bedfordshire
        'Bedford', 'Central Bedfordshire',
        // Cambridgeshire
        'Cambridge', 'East Cambridgeshire', 'Fenland', 'Huntingdonshire', 'South Cambridgeshire',
        // Essex
        'Basildon', 'Braintree', 'Brentwood', 'Castle Point', 'Chelmsford', 'Colchester',
        'Epping Forest', 'Harlow', 'Maldon', 'Rochford', 'Tendring', 'Uttlesford',
        // Hertfordshire
        'Broxbourne', 'Dacorum', 'East Hertfordshire', 'Hertsmere', 'North Hertfordshire',
        'St Albans', 'Stevenage', 'Three Rivers', 'Watford', 'Welwyn Hatfield',
        // Norfolk
        'Broadland', 'Breckland', 'Great Yarmouth', "King's Lynn and West Norfolk",
        'North Norfolk', 'Norwich', 'South Norfolk',
        // Suffolk
        'Babergh', 'Forest Heath', 'Ipswich', 'Mid Suffolk', 'St Edmundsbury',
        'Suffolk Coastal', 'Waveney',
    ],
    'East Midlands': [
        // Unitaries
        'Derby', 'Leicester', 'Nottingham', 'Rutland',
        // Derbyshire
        'Amber Valley', 'Bolsover', 'Chesterfield', 'Derbyshire Dales', 'Erewash',
        'High Peak', 'North East Derbyshire', 'South Derbyshire',
        // Leicestershire
        'Blaby', 'Charnwood', 'Harborough', 'Hinckley and Bosworth', 'Melton',
        'North West Leicestershire', 'Oadby and Wigston',
        // Lincolnshire
        'Boston', 'East Lindsey', 'Lincoln', 'North Kesteven', 'South Holland',
        'South Kesteven', 'West Lindsey',
        // Northamptonshire
        'Corby', 'Daventry', 'East Northamptonshire', 'Kettering', 'Northampton',
        'South Northamptonshire', 'Wellingborough',
        // Nottinghamshire
        'Ashfield', 'Bassetlaw', 'Broxtowe', 'Gedling', 'Mansfield',
        'Newark and Sherwood', 'Rushcliffe',
    ],
    'West Midlands': [
        // Metropolitan boroughs
        'Birmingham', 'Coventry', 'Dudley', 'Sandwell', 'Solihull', 'Walsall', 'Wolverhampton',
        // Unitaries
        'Herefordshire, County of', 'Shropshire', 'Stoke-on-Trent', 'Telford and Wrekin',
        // Staffordshire
        'Cannock Chase', 'East Staffordshire', 'Lichfield', 'Newcastle-under-Lyme',
        'South Staffordshire', 'Stafford', 'Staffordshire Moorlands', 'Tamworth',
        // Warwickshire
        'North Warwickshire', 'Nuneaton and Bedworth', 'Rugby', 'Stratford-on-Avon', 'Warwick',
        // Worcestershire
        'Bromsgrove', 'Malvern Hills', 'Redditch', 'Worcester', 'Wychavon', 'Wyre Forest',
    ],
    'Yorkshire & Humber': [
        // Metropolitan boroughs
        'Barnsley', 'Bradford', 'Calderdale', 'Doncaster', 'Kirklees',
        'Leeds', 'Rotherham', 'Sheffield', 'Wakefield',
        // Unitaries
        'East Riding of Yorkshire', 'Kingston upon Hull, City of', 'York',
        // North Yorkshire districts (2013 boundaries)
        'Craven', 'Hambleton', 'Harrogate', 'Richmondshire', 'Ryedale', 'Scarborough', 'Selby',
        // North Lincolnshire unitaries
        'North East Lincolnshire', 'North Lincolnshire',
    ],
    'North West': [
        // Greater Manchester
        'Bolton', 'Bury', 'Manchester', 'Oldham', 'Rochdale', 'Salford',
        'Stockport', 'Tameside', 'Trafford', 'Wigan',
        // Merseyside
        'Knowsley', 'Liverpool', 'Sefton', 'St. Helens', 'Wirral',
        // Unitaries
        'Blackburn with Darwen', 'Blackpool', 'Cheshire East', 'Cheshire West and Chester',
        'Halton', 'Warrington',
        // Lancashire
        'Burnley', 'Chorley', 'Fylde', 'Hyndburn', 'Lancaster', 'Pendle', 'Preston',
        'Ribble Valley', 'Rossendale', 'South Ribble', 'West Lancashire', 'Wyre',
        // Cumbria (2013 boundaries)
        'Allerdale', 'Barrow-in-Furness', 'Carlisle', 'Copeland', 'Eden', 'South Lakeland',
    ],
    'North East': [
        // Tyne & Wear met boroughs
        'Gateshead', 'Newcastle upon Tyne', 'North Tyneside', 'South Tyneside', 'Sunderland',
        // Unitaries
        'County Durham', 'Darlington', 'Hartlepool', 'Middlesbrough',
        'Northumberland', 'Redcar and Cleveland', 'Stockton-on-Tees',
    ],
}

export const REGION_NAMES = Object.keys(REGIONS)
