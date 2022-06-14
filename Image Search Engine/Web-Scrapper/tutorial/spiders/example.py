# -*- coding: utf-8 -*-
# import the necessary packages
import scrapy
from scrapy.http import Request
from tutorial.items import TutorialItem
from urllib.request import urlretrieve

class ExampleSpider(scrapy.Spider):
    name = 'example'
    allowed_domains = ['www.santabanta.com/']
    start_urls = ['http://www.santabanta.com']
    def parse(self, response):
        self.count = 0
        self.start_urls = ['http://www.santabanta.com']
        Div_Selector = '.dd1-div'
        for i in response.css(Div_Selector):
            NAME_SELECTOR = '.dd1-div1'
            for j in i.css(NAME_SELECTOR):
                Div = ".div-25-percent"
                for k in j.css(Div):
                    href_link = 'li a ::attr(href)'
                    Name_link = 'li a ::text'
                    name = k.css(Name_link).extract_first()
                    if 'Events' in name:
                        href =  k.css(href_link).extract_first()
                        url  = self.start_urls[0] + href
                        self.visited_url = []
                        return Request(url = url, callback = self.parse_page,dont_filter=True)
                    
    def parse_page(self,response):
        self.visited_jpg = [] 
        self.visited_url.append(response.url)
        image_link = '.imgdiv1'
        for i in response.css(image_link):
            href_image = 'a ::attr(href)'
            url = i.css(href_image).extract_first()
            url = self.start_urls[0] + url
            self.visited_url.append(response.url)
            if not any(url in str for str in self.visited_url):
                yield {'First_url': url}
                yield Request(url = url, callback = self.parse_page2)        
                    
        Anchor_tag_href  = response.css('a ::attr(href)').extract() 
        for j in Anchor_tag_href:
            if 'gallery/events/?page' in j:
                url = self.start_urls[0] + j
                if not any(url in str for str in self.visited_url):
                    yield {'Redirect_url': url}
                    yield Request(url = url, callback = self.parse_page, dont_filter=False)
                
                
    def parse_page2(self,response):
        image_link2 = '.imgdiv1'
        for j in response.css(image_link2):
            href_image2 = 'a ::attr(href)'
            url = j.css(href_image2).extract_first()
            url = self.start_urls[0] + url
            if not any(url in str for str in self.visited_url):
                yield {'Second_url': url}
                yield Request(url = url, callback = self.download_image)
    
    def download_image(self,response):
        self.visited_url.append(response.url)
        Meta_tag_img = response.css('meta::attr(content)').extract()
        for k in Meta_tag_img:
            if '.jpg' in k:
                if not any(k in str for str in self.visited_jpg ):
                    self.visited_jpg.append(k)
                    yield {'img': k}
                    #urlretrieve(k, 'G:/Scraped_Output/' + str(self.count) + '.jpg')
                    #self.count += 1
        