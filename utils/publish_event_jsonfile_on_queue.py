#!/usr/bin/env python

import sys
import pika
import json
import argparse


def parseMyLine():

    ll = sys.argv[1:]
    if not ll:
       print ("Type " + sys.argv[0] + " -h or --help for guide and examples")
       sys.exit(0)

    Description = "Create main configuration file for pyPTF code"

    examples    = "Example:\n" + sys.argv[0] + " --rabbit_RK INT.QUAKEEXTPTF.CATDEV --event_jsn test_event.json "

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=Description, epilog=examples)

    parser.add_argument('--event_jsn',       default='None',                     help='json event file. Default=None')
    parser.add_argument('--rabbit_RK',       default='INT.QUAKEEXTPTF.CATDEV',   help='Publishing rabbit-mq RK. Default=INT.QUAKEEXTPTF.CATDEV')
    parser.add_argument('--rabbit_vhost',    default='/rbus',                    help='Events working path name. Default=None')
    parser.add_argument('--rabbit_host',     default='rabbitmq1-rm.int.ingv.it', help='rabbit host servar. Default: rabbitmq1-rm.int.ingv.it')
    parser.add_argument('--rabbit_port',     default='5672',                     help='PTF Data working path name. Default=None')
    parser.add_argument('--rabbit_login',    default='rbus_writer',              help='PTF Data working path name. Default=None')
    parser.add_argument('--rabbit_pass',     default='pass_writer',              help='PTF Data working path name. Default=None')
    parser.add_argument('--rabbit_exchange', default='cat',                      help='PTF Data working path name. Default=None')
    parser.add_argument('--rabbit_type',     default='topic',                    help='PTF Data working path name. Default=None')
    parser.add_argument('--broker',          default='rabbit',                   help='PTF Data working path name. Default=None')


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

    # Read json
    with open(json_to_send, 'r') as file:
        data = json.loads(file.read().replace("\'", "\""))
    print(data)
    #data = json.loads(f.read())



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
                          body = json.dumps(data))


    return True

#### Begin
args=parseMyLine()
nrArguments = sys.argv[1:]
if not nrArguments:
       print ("Use -h or --help option for Help")
       sys.exit(0)


if(args.broker == 'kafka'):
    print("Kafka message broker not jet supported. Exit!")
    sys.exit(0)

sent_via_rabbit = send_contents_via_rabbit(jsn=args.event_jsn,
                                           user_login=args.rabbit_login,
                                           user_password=args.rabbit_pass,
                                           host_server=args.rabbit_host,
                                           port_server=args.rabbit_port,
                                           virtual_host=args.rabbit_vhost,
                                           routing_key=args.rabbit_RK,
                                           exchange_type=args.rabbit_type,
                                           exchange_queue=args.rabbit_exchange)

print("Message %s published on broker %s: %r" % (args.event_jsn, args.broker, sent_via_rabbit))
