from argparse import ArgumentParser
from google.protobuf.json_format import MessageToDict
import rosbag
import json


if __name__ == "__main__":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(dest='bag_path', help='Path to bag.')
    arg_parser.add_argument(dest='save_path', help='Files path.')
    arg_parser.add_argument(dest='topics', nargs='*', help='Topics to convert.')

    args = arg_parser.parse_args()

    # TODO convert all topics
    assert len(args.topics) > 0, "All topics conversion not implemented"

    bag = rosbag.Bag(args.bag_path)
    out_file = open(args.save_path, 'w')

    data = dict({x: [] for x in args.topics})

    for topic, msg, t in bag.read_messages(topics=args.topics):
        dictObj = MessageToDict(msg)
        dictObj["timestamp"] = t.to_sec()
        data[topic].append(dictObj)

    json.dump(data, out_file)

    out_file.close()

    bag.close()