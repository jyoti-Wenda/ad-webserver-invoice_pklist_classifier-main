import time
import os
import re
from stat import *
from celery import Celery
from celery.utils.log import get_task_logger
from utils import AWSutils
from utils import excelutils as xlsutils
from utils import webhookutils as whutils
from utils import layoutLMutils as lmutils
from utils import SMTPutils as smtputils

UPLOAD_FOLDER = '/flask_app/files/'
s3BucketName = "activedocumentsbucket"
s3BucketRegion = "us-east-1"
serviceDirectory = 'activedocuments/bolinstruction_classifier/pdf'
doc_type = 'TEST'
# doc_type = 'BOL'

logger = get_task_logger(__name__)

app = Celery('tasks',
             broker='amqp://admin:mypass@rabbit:5672',
             backend='rpc://')

class perioli_email_set:
    sender = {
        'VGM': [['Wenda Active Documents'], ['vgm@wenda-it.com'], ['qmeeacwfappkmzgn'], ['smtp.gmail.com']],
        'BOL': [['Wenda Active Documents'], ['istruzioni@wenda-it.com'], ['slgugfsjkyvfzday'], ['smtp.gmail.com']],
        'MAN': [['Wenda Active Documents'], ['manifesti@wenda-it.com'], ['nrlpfzjiojsendky'], ['smtp.gmail.com']],
        'TEST': [['Wenda Active Documents'], ['istruzioni@wenda-it.com'], ['slgugfsjkyvfzday'], ['smtp.gmail.com']]
        }

    receiver = {
        'VGM': [['Federico Natale','Giulia Rossi'], ['f.natale@cnanitalia.it','giuliarossi@darioperioli.it']],
        'BOL': [['Noemi Valerio','L. Pardini','Perioli Spedizioni'], ['n.valerio@cnanitalia.it','l.pardini@cnanitalia.it','spedizioni@darioperioli.it']],
        'MAN': [],
        'TEST': [['Valentina Protti'], ['valentina@wenda-it.com']]
        # 'TEST': [['Valentina Protti','Luca Boarini'], ['valentina@wenda-it.com','luca@wenda-it.com']]
        }

# app.conf.broker_pool_limit = 0
app.conf.task_publish_retry = False
app.conf.broker_heartbeat = 1800


@app.task()
def elab_file(user, filePath, save, output, ocr, webhook, pathfile, localpath):
    logger.info('ASYNC POST /uploader > Got Request - Starting work')
    print(filePath)
    if os.path.isfile(filePath):
        print('I can read this file path!')
        if save:
            AWSutils.uploadToBucket(s3BucketName, s3BucketRegion, serviceDirectory, UPLOAD_FOLDER, user, os.path.basename(filePath))
        formatted_result, result = lmutils.elab(filePath, ocr)
        if output.lower() == 'excel':
            logger.info('Sending the classified doc to their elab endpoints')
            task_ids = lmutils.send_to_elaboration(filePath, result, webhook, pathfile, localpath, output, doc_type)
            logger.info(task_ids)
            # if needed we can also collect the results from the model extraction and send them back here
            return task_ids, result, 'json'
        else:
            logger.info('Work Finished')
            return formatted_result, result, 'json'
    logger.info('Work Finished')
    return filePath, os.path.basename(filePath), 'pdf'