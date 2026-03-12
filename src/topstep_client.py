"""
Topstep (ProjectX) API client with authentication.
API docs: https://gateway.docs.projectx.com/
"""

import os
from datetime import datetime, timezone
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.topstepx.com"


class TopstepClient:
    def __init__(self, username: str = None, api_key: str = None):
        self.username = username or os.environ["TOPSTEP_USERNAME"]
        self.api_key = api_key or os.environ["TOPSTEP_API_KEY"]
        self.token: str | None = None
        self._client = httpx.Client(base_url=BASE_URL, timeout=60)

    def login(self) -> str:
        """Authenticate and store the session token. Returns the token."""
        resp = self._client.post(
            "/api/Auth/loginKey",
            json={"userName": self.username, "apiKey": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            raise RuntimeError(f"Login failed: {data.get('errorCode')} – {data.get('errorMessage')}")

        self.token = data["token"]
        self._client.headers["Authorization"] = f"Bearer {self.token}"
        return self.token

    def get_accounts(self) -> list[dict]:
        """Return all active practice/combine accounts."""
        self._ensure_authenticated()
        resp = self._client.post(
            "/api/Account/search",
            json={"onlyActiveAccounts": True},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Account search failed: {data.get('errorMessage')}")
        return data.get("accounts", [])

    # Bar units
    SECOND = 1
    MINUTE = 2
    HOUR   = 3
    DAY    = 4
    WEEK   = 5
    MONTH  = 6

    def search_contracts(self, text: str, live: bool = False) -> list[dict]:
        """Search contracts by name (e.g. 'NQ', 'ES'). Returns up to 20 results."""
        self._ensure_authenticated()
        resp = self._client.post(
            "/api/Contract/search",
            json={"searchText": text, "live": live},
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"Contract search failed: {data.get('errorMessage')}")
        return data.get("contracts", [])

    def get_bars(
        self,
        contract_id: int,
        start: datetime,
        end: datetime,
        unit: int = 2,       # MINUTE
        unit_number: int = 1,
        limit: int = 1000,
        live: bool = False,
        include_partial: bool = False,
    ) -> list[dict]:
        """
        Retrieve historical OHLCV bars for a contract.

        Returns a list of dicts with keys: t, o, h, l, c, v
          t = timestamp (ISO string), o/h/l/c = prices, v = volume

        unit: TopstepClient.SECOND/MINUTE/HOUR/DAY/WEEK/MONTH
        limit: max bars returned (capped at 20,000 per request)
        """
        self._ensure_authenticated()

        def _fmt(dt: datetime) -> str:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()

        resp = self._client.post(
            "/api/History/retrieveBars",
            json={
                "contractId": contract_id,
                "live": live,
                "startTime": _fmt(start),
                "endTime": _fmt(end),
                "unit": unit,
                "unitNumber": unit_number,
                "limit": limit,
                "includePartialBar": include_partial,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise RuntimeError(f"retrieveBars failed: {data.get('errorMessage')}")
        return data.get("bars", [])

    def _ensure_authenticated(self):
        if not self.token:
            self.login()

    def __enter__(self):
        self.login()
        return self

    def __exit__(self, *_):
        self._client.close()


if __name__ == "__main__":
    from datetime import timedelta

    with TopstepClient() as client:
        # Find the front-month NQ contract
        contracts = client.search_contracts("MES")
        if not contracts:
            print("No NQ contracts found.")
        else:
            contract = contracts[0]
            print(f"Contract: {contract['name']}  id={contract['id']}")

            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=4)

            bars = client.get_bars(
                contract_id=contract["id"],
                start=start,
                end=end,
                unit=TopstepClient.MINUTE,
                unit_number=5,   # 5-minute bars
                limit=100,
            )
            print(f"Fetched {len(bars)} 5-min bars")
            for bar in bars[-5:]:  # print last 5
                print(f"  {bar.get('t')}  o={bar.get('o')}  h={bar.get('h')}  l={bar.get('l')}  c={bar.get('c')}  v={bar.get('v')}")
