MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

EXTRACTION_PROMPT = """
You are an expert OCR-to-JSON extractor. 
Extract and structure all relevant fields from the document into the following strict JSON format:

```json
{
  "shippingBillNumber": "",
  "shippingBillDate": "",
  "invoiceNumber": "",
  "invoiceDate": "",
  "portCode": "",
  "location": "",
  "items": [
    {
      "itemNumber": "",
      "quantity": "",
      'itemDescription': "",
      "billOfEntry": [
        {
          "billOfEntryNumber": "",
          "billOfEntryDate": ""
        }
      ]
    }
  ]
}
```

### Rules:
- Never return null values; use empty string "" for missing text, and [] for missing lists.
- `itemNumber` must be the **8-digit RITC Code** if available.
- `quantity` should include units (e.g., "192.000NOS").
- `billOfEntry` may contain multiple entries (billOfEntryNumber, billOfEntryDate). If none, return [].
- Always wrap the output in valid JSON strictly following the schema above.
- PortCode will be in either exporter address of 6 digits or letters or mix or will be mentioned as "Port Code: portcode value".
"""