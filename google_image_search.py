from serpapi import GoogleSearch

params = {
  "engine": "google_reverse_image",
  "api_key": "3d585e4ed0198f5118ce70252edef2ff6d49a8d2ad319da4d7e663a337be333b",
  "location_requested": "Hanoi, Vietnam",
  "location_used": "Hanoi,Vietnam",
  "google_domain": "google.com.vn",
  "hl": "vi",
  "gl": "vn",
  "device": "desktop"
}

def google_reverse_image(url):
    params["image_url"] = url
    search = GoogleSearch(params)
    results = search.get_dict()
    inline_images = results["inline_images"]
    return (inline_images)