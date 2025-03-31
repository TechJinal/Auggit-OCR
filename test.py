import json
import re


# ✅ Merge JSON Data Logic
def combine_page_data(data):
    """Combine all key-value pairs from multiple pages into a single JSON."""
    combined_data = {
        "shippingBillNumber": None,
        "invoiceNumber": None,
        "shippingBillDate": None,
        "invoiceDate": None,
        "portCode": None,
        "location": None,
        "items": []
    }

    for page_data in data.values():
        # Assign header-level fields if not already set
        for key in ["shippingBillNumber", "invoiceNumber", "shippingBillDate", "invoiceDate", "portCode", "location"]:
            if combined_data[key] is None and page_data.get(key):
                combined_data[key] = page_data[key]

        # Get item-level fields (check if None before processing)
        item_numbers = page_data.get("itemNumber", []) or []
        quantities = page_data.get("quantity", []) or []
        item_details = page_data.get("itemDetails", []) or []

        # ✅ Filter valid 8-digit item numbers only
        valid_item_numbers = [item for item in item_numbers if item and re.match(r"^\d{8}$", str(item))]

        # ✅ Check length consistency before adding items
        if valid_item_numbers and quantities and item_details:
            for i in range(len(valid_item_numbers)):
                if i < len(quantities) and i < len(item_details):
                    item = {
                        "itemNumber": valid_item_numbers[i],
                        "quantity": quantities[i],
                        "itemDetails": item_details[i]
                    }
                    # Add item to the final list
                    combined_data["items"].append(item)

    # ✅ Remove empty or null values
    for key, value in combined_data.items():
        if isinstance(value, list) and not value:
            combined_data[key] = None

    return combined_data


# ✅ Combine data from all pages
# final_combined_data = combine_page_data(data)
# print("Final Combined Data:", final_combined_data)

# ✅ Print final merged JSON
# print(json.dumps(final_combined_data, indent=4))
