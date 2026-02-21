export const RISK = {
    low:    { colour: '#6b8f5e', label: 'Low risk'  },
    medium: { colour: '#c8962a', label: 'Med. risk' },
    high:   { colour: '#b85c38', label: 'High risk' },
}

export type RiskLevel = keyof typeof RISK

export interface Report {
    id:       number
    title:    string
    location: string
    date:     string
    snippet:  string
    risk:     RiskLevel
    type:     string
}

export const REPORTS: Report[] = [
    {
        id:       1,
        title:    'Mid-Scale Residential — Top Performing Councils',
        location: 'National Overview',
        date:     '14 Feb 2026',
        snippet:  'Regional approval rates exceed the national mean by 14%. Median determination time: 9.2 weeks.',
        risk:     'low',
        type:     'Residential',
    },
    {
        id:       2,
        title:    'High-Rise Commercial — Risk Exposure Report',
        location: 'Greater London',
        date:     '12 Feb 2026',
        snippet:  'Refusal rates increase sharply in boroughs with active Article 4 directions. Conservation area coverage exceeds 40% across 6 of the 8 highest-risk LPAs.',
        risk:     'high',
        type:     'Commercial',
    },
    {
        id:       3,
        title:    'Change of Use — Regional Variance',
        location: 'South East England',
        date:     '10 Feb 2026',
        snippet:  'Permitted development conversion rates vary by 31% across sub-regions. Non-metropolitan LPAs resolve applications 3.1 weeks faster on average.',
        risk:     'medium',
        type:     'Change of Use',
    },
]
