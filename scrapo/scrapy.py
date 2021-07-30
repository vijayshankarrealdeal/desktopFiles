import scrapy

class Scrapy(scrapy.Spider):
    name = "books"
    start_url = [
        
        "https://books.toscrape.com/",
        "https://books.toscrape.com/catalogue/category/books/fantasy_19/index.html",
        
        ]

    def parse(self,response):
        page = response.url.split('/')[-2]
        filename = 'books-%s' % page
        
    