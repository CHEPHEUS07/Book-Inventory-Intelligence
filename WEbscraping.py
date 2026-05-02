"""
Web Scraping Project - CodeAlpha Data Analytics Internship (Task 1)
===================================================================
This script scrapes book data from books.toscrape.com using BeautifulSoup.
It collects book titles, prices, ratings, and availability across multiple pages
and saves the data to a CSV file for further analysis.

Libraries Used: requests, BeautifulSoup (bs4), csv
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import time

# Base URL of the website to scrape
BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

def get_star_rating(element):
    """Convert word-based star rating to a number."""
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }
    star_class = element.find("p", class_="star-rating")
    if star_class:
        # The rating is stored as a CSS class name (e.g., "star-rating Three")
        rating_word = star_class["class"][1]
        return rating_map.get(rating_word, 0)
    return 0


def scrape_book_details(book_url):
    """Scrape additional details from an individual book's page."""
    try:
        response = requests.get(book_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Get book description
        description_tag = soup.find("meta", attrs={"name": "description"})
        description = description_tag["content"].strip() if description_tag else "No description available"

        # Get product information from the table
        table = soup.find("table", class_="table-striped")
        product_info = {}
        if table:
            rows = table.find_all("tr")
            for row in rows:
                header = row.find("th").text.strip()
                value = row.find("td").text.strip()
                product_info[header] = value

        category_tag = soup.find("ul", class_="breadcrumb")
        category = "Unknown"
        if category_tag:
            links = category_tag.find_all("a")
            if len(links) >= 3:
                category = links[2].text.strip()

        return {
            "description": description[:200],  # Limit description length
            "upc": product_info.get("UPC", "N/A"),
            "product_type": product_info.get("Product Type", "N/A"),
            "price_excl_tax": product_info.get("Price (excl. tax)", "N/A"),
            "price_incl_tax": product_info.get("Price (incl. tax)", "N/A"),
            "tax": product_info.get("Tax", "N/A"),
            "availability": product_info.get("Availability", "N/A"),
            "num_reviews": product_info.get("Number of reviews", "0"),
            "category": category
        }
    except Exception as e:
        print(f"  ⚠ Error fetching book details: {e}")
        return None


def scrape_books(num_pages=5):
    """
    Scrape book data from multiple pages of books.toscrape.com.
    
    Args:
        num_pages (int): Number of pages to scrape (max 50)
    
    Returns:
        list: A list of dictionaries containing book data
    """
    all_books = []

    print("=" * 60)
    print("  WEB SCRAPING - books.toscrape.com")
    print("  CodeAlpha Data Analytics Internship - Task 1")
    print("=" * 60)
    print()

    for page in range(1, num_pages + 1):
        url = BASE_URL.format(page)
        print(f"📄 Scraping page {page}/{num_pages}: {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error fetching page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.find_all("article", class_="product_pod")

        if not books:
            print(f"  ⚠ No books found on page {page}. Stopping.")
            break

        for book in books:
            # Extract basic info from the listing page
            title_tag = book.find("h3").find("a")
            title = title_tag["title"]
            relative_url = title_tag["href"]
            book_url = "https://books.toscrape.com/catalogue/" + relative_url.replace("../", "")

            price = book.find("p", class_="price_color").text.strip()
            rating = get_star_rating(book)
            in_stock = "In stock" if book.find("p", class_="instock") else "Out of stock"

            # Fetch detailed info from the book's individual page
            print(f"  📖 Scraping: {title[:50]}...")
            details = scrape_book_details(book_url)

            book_data = {
                "Title": title,
                "Price": price,
                "Rating (out of 5)": rating,
                "Stock Status": in_stock,
                "Book URL": book_url,
            }

            if details:
                book_data.update({
                    "Category": details["category"],
                    "UPC": details["upc"],
                    "Price (excl. tax)": details["price_excl_tax"],
                    "Price (incl. tax)": details["price_incl_tax"],
                    "Tax": details["tax"],
                    "Availability": details["availability"],
                    "Number of Reviews": details["num_reviews"],
                    "Description": details["description"],
                })

            all_books.append(book_data)

            # Small delay to be polite to the server
            time.sleep(0.3)

        print(f"  ✅ Scraped {len(books)} books from page {page}")
        print()

    return all_books


def save_to_csv(data, filename):
    """Save the scraped data to a CSV file."""
    if not data:
        print("❌ No data to save!")
        return

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Data saved to: {filepath}")
    print(f"   Total records: {len(data)}")
    print(f"   Columns: {', '.join(fieldnames)}")


# ===========================
#   MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    # Scrape 3 pages (60 books) - adjust num_pages for more/less data
    books_data = scrape_books(num_pages=3)

    # Save to CSV
    save_to_csv(books_data, "scraped_books_data.csv")

    print()
    print("=" * 60)
    print("  ✅ WEB SCRAPING COMPLETE!")
    print(f"  📊 Total books scraped: {len(books_data)}")
    print("  📁 Output file: scraped_books_data.csv")
    print("=" * 60)