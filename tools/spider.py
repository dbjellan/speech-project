import urlparse
import os

from scrapy.http import Request
from scrapy.spider import BaseSpider

project_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(project_directory, 'corpus')):
    os.mkdir(os.path.join(project_directory, 'corpus'))
save_directory = os.path.join(project_directory, 'corpus/download')
if not os.path.exists(save_directory):
    os.mkdir(save_directory)
extract_directory = os.path.join(project_directory, 'corpus/extracted')
if not os.path.exists(extract_directory):
    os.mkdir(extract_directory)
os.chdir(extract_directory)


class VoxSpeechSpider(BaseSpider):
    name = "voxspider"
    start_urls = ['http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/']

    def parse(self, response):
        base_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
        for a in response.xpath('//a[@href]/@href'):
            link = a.extract()
            if link.endswith('.tgz'):
                link = urlparse.urljoin(base_url, link)
                yield Request(link, callback=self.save_file)

    def save_file(self, response):
        name = response.url.split('/')[-1]
        print('Downloading and extraction: %s' % (name, ))
        path = os.path.join(save_directory, name)
        with open(path, 'wb') as f:
            f.write(response.body)
        os.system('tar -xvf ../download/%s' % (name,))

