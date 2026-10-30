"""Freeze official Sydney on-demand Nova Lite public prices and SKU evidence."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import requests

from rag_lab.io import write_json

url = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/ap-southeast-2/index.json"
response = requests.get(url, timeout=60)
response.raise_for_status()
offer = response.json()
price = {
    "model_id": "amazon.nova-lite-v1:0",
    "region": "ap-southeast-2",
    "verified_at": datetime.now(UTC).isoformat(),
    "source_url": "https://aws.amazon.com/bedrock/pricing/",
    "price_list_url": url,
    "publication_date": offer["publicationDate"],
    "evidence": {},
}
for sku, product in offer["products"].items():
    attrs = product["attributes"]
    if (
        attrs.get("model") != "Nova Lite"
        or attrs.get("feature") != "On-demand Inference"
        or attrs.get("inferenceType") not in ["Input tokens", "Output tokens"]
    ):
        continue
    key = (
        "input_usd_per_million"
        if attrs["inferenceType"] == "Input tokens"
        else "output_usd_per_million"
    )
    if key in price:
        raise ValueError("Ambiguous on-demand price")
    terms = list(offer["terms"]["OnDemand"][sku].values())
    dimensions = list(terms[0]["priceDimensions"].values())
    if len(terms) != 1 or len(dimensions) != 1 or dimensions[0]["unit"] != "1K tokens":
        raise ValueError("Unexpected official pricing dimensions")
    price[key] = float(Decimal(dimensions[0]["pricePerUnit"]["USD"]) * 1000)
    price["evidence"][key] = {"product": product, "term": terms[0]}
if not all(k in price for k in ["input_usd_per_million", "output_usd_per_million"]):
    raise ValueError("Missing regional price")
write_json(Path("configs/pricing.local.json"), price)
print({k: v for k, v in price.items() if k != "evidence"})
