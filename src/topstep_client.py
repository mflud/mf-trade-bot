"""
Topstep (ProjectX) API client with authentication.
API docs: https://gateway.docs.projectx.com/
"""

import os
from datetime import datetime, timezone
from typing import Literal
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

    # Known MES contracts in chronological order, oldest first.
    # Extend this list when new quarterly contracts become available.
    MES_CONTRACTS = [
        "CON.F.US.MES.H25",  # Mar 2025
        "CON.F.US.MES.M25",  # Jun 2025
        "CON.F.US.MES.U25",  # Sep 2025
        "CON.F.US.MES.Z25",  # Dec 2025
        "CON.F.US.MES.H26",  # Mar 2026
    ]

    def get_continuous_mes_bars(
        self,
        unit: int = 4,           # DAY
        unit_number: int = 1,
        limit: int = 20000,
        back_adjust: bool = True,
    ) -> list[dict]:
        """
        Return a stitched continuous MES bar series across all available quarterly contracts.

        On overlapping dates, the newer (next-quarter) contract takes precedence.
        With back_adjust=True (default), historical prices are shifted at each roll so
        that the series is gap-free (Panama/backward method). Prices reflect returns
        accurately but are not actual traded prices.
        With back_adjust=False, raw prices are returned with visible roll gaps.

        Returns list of dicts sorted ascending by timestamp, each with:
          t, o, h, l, c, v, contract (source contract id)
        """
        self._ensure_authenticated()

        far_past = datetime(2020, 1, 1, tzinfo=timezone.utc)
        far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)

        # Fetch all contracts; skip those with no data
        per_contract: list[list[dict]] = []
        for cid in self.MES_CONTRACTS:
            bars = self.get_bars(
                contract_id=cid,
                start=far_past,
                end=far_future,
                unit=unit,
                unit_number=unit_number,
                limit=limit,
            )
            if not bars:
                continue
            # API returns newest-first; reverse to ascending
            bars = list(reversed(bars))
            for b in bars:
                b["contract"] = cid
            per_contract.append(bars)

        if not per_contract:
            return []

        # Merge: newer contract wins on any overlapping timestamp
        # Build a dict keyed by timestamp, iterating oldest→newest contract
        merged: dict[str, dict] = {}
        for bars in per_contract:
            for b in bars:
                merged[b["t"]] = b

        series = sorted(merged.values(), key=lambda b: b["t"])

        if not back_adjust:
            return series

        # Back-adjust (Panama method, backward pass):
        # At each contract switch, calculate price gap and subtract it from all
        # earlier bars so the series is continuous at the roll point.
        cumulative_adj = 0.0
        prev_contract = series[-1]["contract"]

        # Walk backward; when contract changes, compute gap and accumulate
        for i in range(len(series) - 2, -1, -1):
            cur = series[i]
            nxt = series[i + 1]

            if nxt["contract"] != prev_contract:
                # Roll: gap = close of old contract day - close of new contract day
                # (nxt is the first bar of the new contract, cur is last of old)
                gap = cur["c"] - nxt["c"]
                cumulative_adj += gap
                prev_contract = nxt["contract"]

            if cumulative_adj != 0.0:
                for field in ("o", "h", "l", "c"):
                    cur[field] = round(cur[field] + cumulative_adj, 4)

        return series

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
