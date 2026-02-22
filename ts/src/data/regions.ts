/**
 * Region data has been moved to the backend.
 * Fetch it via GET /api/councils — see src/api/client.ts → fetchCouncils().
 *
 * Canonical region ordering (used to sort the dropdown).
 */
export const REGION_ORDER = [
  'London',
  'South East',
  'South West',
  'East of England',
  'East Midlands',
  'West Midlands',
  'Yorkshire & Humber',
  'North West',
  'North East',
] as const
