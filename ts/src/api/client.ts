export interface QueryResponse {
  boroughs: Record<string, number>
  summary: string
}

const DUMMY_RESPONSE: QueryResponse = {
  boroughs: {
    'Barnet': 82,
    'Camden': 45,
    'Hackney': 63,
    'Islington': 38,
    'Tower Hamlets': 71,
    'Southwark': 56,
    'Lambeth': 49,
    'Westminster': 22,
    'Kensington and Chelsea': 15,
    'Hammersmith and Fulham': 34,
    'Wandsworth': 67,
    'Lewisham': 74,
    'Greenwich': 78,
    'Bexley': 85,
    'Bromley': 88,
    'Croydon': 79,
    'Merton': 61,
    'Kingston upon Thames': 72,
    'Richmond upon Thames': 41,
    'Hounslow': 58,
    'Ealing': 65,
    'Hillingdon': 76,
    'Harrow': 69,
    'Brent': 53,
    'Haringey': 47,
    'Enfield': 81,
    'Waltham Forest': 70,
    'Redbridge': 73,
    'Havering': 84,
    'Barking and Dagenham': 77,
    'Newham': 66,
    'Birmingham': 55,
    'Manchester': 62,
    'Leeds': 68,
    'Sheffield': 59,
    'Bristol, City of': 52,
    'Liverpool': 43,
    'Newcastle upon Tyne': 57,
    'Nottingham': 64,
    'Leicester': 48,
    'Coventry': 51,
    'Bradford': 73,
    'Sunderland': 80,
    'Wolverhampton': 39,
    'Plymouth': 71,
    'Southampton': 54,
    'Reading': 60,
    'Derby': 46,
    'Stoke-on-Trent': 83,
    'Swindon': 75,
  },
  summary: 'Based on your property development idea, we analysed planning permission approval rates across English boroughs. Green areas have the highest likelihood of approval.',
}

export async function submitQuery(_message: string): Promise<QueryResponse> {
  // Simulate network delay
  await new Promise((resolve) => setTimeout(resolve, 800))
  return DUMMY_RESPONSE
}
