import re

def merge_data(data):
    merged_data = {
        "shippingBillNumber": "",
        "invoiceNumber": "",
        "shippingBillDate": "",
        "invoiceDate": "",
        "portCode": "",
        "location": "",
        "items": []
    }
    
    # Extract common metadata from the first non-empty page
    for page in data.values():
        for key in ["shippingBillNumber", "invoiceNumber", "shippingBillDate", "invoiceDate", "portCode", "location"]:
            if not merged_data[key] and page.get(key):
                merged_data[key] = page[key]
    
    # Collect item details
    all_items = []
    for page in data.values():
        item_numbers = page.get("itemNumber", [])
        quantities = page.get("quantity", [])
        item_details = page.get("itemDetails", [])
        
        # Ensure all lists have the same length by padding with empty strings
        max_length = max(len(item_numbers), len(quantities), len(item_details))
        item_numbers += [""] * (max_length - len(item_numbers))
        quantities += [""] * (max_length - len(quantities))
        item_details += [""] * (max_length - len(item_details))
        
        all_items.extend(
            {"itemNumber": item_numbers[i], "quantity": quantities[i], "itemDetails": item_details[i]}
            for i in range(max_length)
        )
    
    # Filter out items where itemNumber does not have exactly 8 characters
    merged_data["items"] = all_items

    return merged_data

def filter_items(data):
    data["items"] = [
        item for item in data["items"]
        if item["itemNumber"] and re.fullmatch(r"\d{8}", item["itemNumber"])
    ]
    return data