import obd
import time

connection = obd.Async(fast=False)

obd.logger.setLevel(obd.logging.DEBUG)  # enables all debug information


# a callback that prints every new value to the console
def new_rpm(r):
    print r.value

connection.watch(obd.commands.RPM, callback=new_rpm)
connection.start()

# the callback will now be fired upon receipt of new values

time.sleep(60)
connection.stop()
