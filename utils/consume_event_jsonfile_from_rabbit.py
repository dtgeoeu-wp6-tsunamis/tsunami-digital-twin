#!/usr/bin/env python

import sys
import pika
import argparse
import getpass
import socket
from time import gmtime, strftime
from tornado          import escape

class CNTClient(object):

      def __init__(self, host, vhost, port, user, password, exchange_type, exchange_queue, binding_key, exchange_key, queue_name, mode):

          self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                            host = host,
                            virtual_host = vhost,
                            port = port,
                            credentials = pika.PlainCredentials(user, password)))

          self.channel = self.connection.channel()

          if(mode == 'clean'):
             self.result  = self.channel.queue_purge(queue=queue_name)

          self.result  = self.channel.queue_declare(queue = queue_name, exclusive = False, durable = True)

          queue_name = self.result.method.queue

          self.callback_queue = self.result.method.queue

          self.channel.queue_bind(exchange=exchange_queue,
                                  queue=queue_name,
                                  routing_key=binding_key)

          #self.channel.queue_bind(exchange=exchange_queue,
          #                        queue=queue_name,
          #                        routing_key="INT.CAT.LOG")
          #
          #self.channel.queue_bind(exchange=exchange_queue,
          #                        queue=queue_name,
          #                        routing_key="INT.DPCHBT.VER")



def parseMyLine():

    ll = sys.argv[1:]
    if not ll:
       print ("Type " + sys.argv[0] + " -h or --help for guide and examples")
       sys.exit(0)

    Description = "Create main configuration file for pyPTF code"

    examples    = "Example:\n" + sys.argv[0] + " --rabbit_RK INT.QUAKEEXTPTF.CATDEV "

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)

    #parser.add_argument('--event_jsn',         default='None',                     help='json event file. Default=None')
    parser.add_argument('--rabbit_RK',         default=None,                       help='Publishing rabbit-mq RK. Default=INT.QUAKEEXTPTF.CATDEV')
    parser.add_argument('--rabbit_vhost',      default='/rbus',                    help='rabbit virtual host. Default=/rbus')
    parser.add_argument('--rabbit_host',       default='rabbitmq1-rm.int.ingv.it', help='rabbit host server. Default: rabbitmq1-rm.int.ingv.it')
    parser.add_argument('--rabbit_port',       default='5672',                     help='rabbit port number. Default=5672')
    parser.add_argument('--rabbit_login',      default='rbus_writer',              help='rabbit login. Default=rbus_writer')
    parser.add_argument('--rabbit_pass',       default='pass_writer',              help='rabbit password. Default=pass_writer')
    parser.add_argument('--rabbit_exchange',   default='cat',                      help='rabbit exchange. Default=cat')
    parser.add_argument('--rabbit_type',       default='topic',                    help='rabbit type. Default=topic')
    parser.add_argument('--broker',            default='rabbit',                   help='broker [rabbit]/kafka')
    parser.add_argument('--rabbit_consumer_q', default='test_ptf_3cards',          help='rabbit consumer queue. Default=test_ptf_3cards')
    parser.add_argument('--rabbit_mode',       default='clean',                    help='service start mode. save: hold and process the queue. clean: empty queue before starting. Default=clean')


    args=parser.parse_args()

    return args

def send_contents_via_rabbit(**kwargs):

    json_to_send   = kwargs.get('jsn', None)
    user_login     = kwargs.get('user_login', None)
    user_password  = kwargs.get('user_password', None)
    host_server    = kwargs.get('host_server', None)
    port_server    = kwargs.get('port_server', None)
    virtual_host   = kwargs.get('virtual_host', None)
    routing_key    = kwargs.get('routing_key', None)
    exchange_queue = kwargs.get('exchange_queue', None)
    exchange_type  = kwargs.get('exchange_type', None)


    user_credentials = pika.PlainCredentials(user_login, user_password)

    try:
      connection = pika.BlockingConnection(pika.ConnectionParameters(
                   connection_attempts = 10,
                   socket_timeout      = 1,
                   retry_delay         = 2,
                   host                = host_server,
                   port                = int(port_server),
                   virtual_host        = virtual_host,
                   credentials         = user_credentials))
    except:
                   print ("Can't open connection to rabbit server: " + host_server)
                   print ("                                port:   " + str(port_server))
                   print ("                                vhost:  " + virtual_host)
                   print ("                                user:   " + user_login)
                   print ("                                pass:   " + user_password)
                   return False


    # Initializs channel connection
    channel = connection.channel()
    print("ACTION: Connecting to rabbitmq with routing_key   :           ", routing_key)
    print("                               with exchange_queue:           ", exchange_queue)
    print("                               with exchange_type :           ", exchange_type)


    # Define channel type queue
    channel.exchange_declare(exchange = exchange_queue,
                             exchange_type = exchange_type,
                             durable = True)

    #
    channel.basic_publish(exchange = exchange_queue,
                          routing_key = routing_key,
                          body = json_to_send)


    return True

def pass2procedure(ch, method, properties, in_json):

    time_connection = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    print("Message json found on RK:", method.routing_key, 'at time:', time_connection)
    jsn_object = escape.json_decode(in_json)
    print(jsn_object)

#### Begin
args=parseMyLine()
nrArguments = sys.argv[1:]
if not nrArguments:
       print ("Use -h or --help option for Help")
       sys.exit(0)



#try:
client=CNTClient(host           = args.rabbit_host,
                 vhost          = args.rabbit_vhost,
                 port           = args.rabbit_port,
                 user           = args.rabbit_login,
                 password       = args.rabbit_pass,
                 exchange_type  = args.rabbit_type,
                 exchange_queue = args.rabbit_exchange,
                 binding_key    = args.rabbit_RK,
                 exchange_key   = args.rabbit_RK,
                 queue_name     = args.rabbit_consumer_q,
                 mode           = args.rabbit_mode)


client.channel.basic_consume(client.callback_queue,
                                pass2procedure,
                                auto_ack=True)

try:
   aa=client.channel.start_consuming()
except KeyboardInterrupt:
   client.channel.stop_consuming()
   client.connection.close()
   print ("Listener break down at time ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
