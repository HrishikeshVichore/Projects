# -*- coding: utf-8 -*-
import scrapy
from scrapy.http import Request
from NasaImages.items import NasaimagesItem

class GetimagesSpider(scrapy.Spider):
    name = 'GetImages'
    allowed_domains = ['https://apod.nasa.gov/apod/archivepix.html']
    start_urls = ['https://apod.nasa.gov/apod/archivepix.html']
    def parse(self, response):
        self.url = ['https://apod.nasa.gov/apod/']
        B_selector = 'b'
        for i in response.css(B_selector):
            href_link = 'a ::attr(href)'
            for j in i.css(href_link).extract():
                url1 = self.url[0] + j
                yield Request(url = url1, callback = self.download_image)
    def download_image(self,response):
        A_selector = 'a ::attr(href)'
        image_extension = ['.jpg','.jpeg','.gif','.tif','.png']
        for k in response.css(A_selector):
            img = k.extract()
            for ie in image_extension:
                if ie in img:
                    if 'http' in img:
                        name = img.split('/')[-1]
                        yield {'img': img}
                    else:    
                        img1 = self.url[0] + img
                        name = img.split('/')[-1]
                        yield {'img': img1}
        
