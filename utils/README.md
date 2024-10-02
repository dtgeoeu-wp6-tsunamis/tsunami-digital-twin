##### Run the consumer on a terminal in order to check the published message

 `./consume_event_jsonfile_from_rabbit.py --rabbit_RK INT.QUAKEEXTPTF.CATDEV`\

##### Publish the json file on a different terminal
termoinal 2: `./publish_event_jsonfile_on_queue.py --rabbit_RK INT.QUAKEEXTPTF.CATDEV --event_jsn test_event.json`
