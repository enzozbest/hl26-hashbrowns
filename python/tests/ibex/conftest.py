"""Shared test fixtures for Ibex connector tests."""
import pytest
import respx
import httpx


BASE_URL = "https://ibex.seractech.co.uk"

TEST_API_KEY = "test-jwt-token"


@pytest.fixture
def mock_ibex():
    """respx mock transport at transport level for IbexClient."""
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as mock:
        yield mock


@pytest.fixture
def search_response_one_result():
    """Single Camden result matching the OpenAPI spec example."""
    return [
        {
            "council_id": 240,
            "council_name": "Camden",
            "planning_reference": "2025/0970/P",
            "url": "https://planningrecords.camden.gov.uk/example",
            "proposal": "Installation of paving and widening of gate.",
            "geometry": "POINT(528349 186246)",
            "raw_address": "6a Oak Court London NW5 1QU",
            "raw_application_type": "Full Planning Permission",
            "normalised_application_type": "full planning application",
            "application_date": "2025-04-01",
            "decided_date": "2025-05-08",
            "raw_decision": "Granted",
            "normalised_decision": "Approved",
        }
    ]


@pytest.fixture
def applications_response_fixture():
    """Rochdale application matching the OpenAPI spec example."""
    return [
        {
            "council_id": 10,
            "council_name": "Rochdale",
            "planning_reference": "24/00057/FUL",
            "url": "https://publicaccess.rochdale.gov.uk/example",
            "proposal": "Alterations to shop front.",
            "raw_address": "119 Yorkshire Street Rochdale OL16 1DS",
            "raw_application_type": "Full Planning Application",
            "application_date": "2025-04-10T00:00:00.000Z",
            "decided_date": "2025-08-08T00:00:00.000Z",
            "normalised_application_type": "full planning application",
            "normalised_decision": "Approved",
            "raw_decision": "Grant subject to conditions",
            "geometry": "POLYGON((389887.2 413707.05,389882.9 413704.2,389887.2 413707.05))",
        }
    ]


@pytest.fixture
def stats_response_fixture():
    """Stats response matching the OpenAPI StatsResponseSchema."""
    return {
        "council_development_activity_level": "medium",
        "approval_rate": 85.5,
        "refusal_rate": 14.5,
        "average_decision_time": {
            "small residential": 45,
            "tree": 28,
            "large residential": 120,
            "home improvement": 35,
            "mixed": 65,
            "medium residential": 75,
        },
        "number_of_applications": {
            "non-material amendment": 150,
            "discharge of conditions": 200,
            "listed building consent": 45,
            "advertisement consent": 80,
            "householder planning application": 500,
            "tree preservation order": 120,
            "lawful development": 90,
            "change of use": 60,
            "full planning application": 350,
            "conservation area": 30,
            "utilities": 25,
            "unknown": 10,
            "environmental impact": 5,
            "section 106": 15,
            "pre-application": 100,
            "other": 40,
        },
        "number_of_new_homes_approved": 250,
    }
