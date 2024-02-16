# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import Jingle.utility.grpc.protogen.utility_update_pb2 as utility__update__pb2


class UtilityMessagingStub(object):
    """Interface
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.PublishUtility = channel.unary_unary(
                '/Jingle.UtilityMessaging/PublishUtility',
                request_serializer=utility__update__pb2.UtilityMessage.SerializeToString,
                response_deserializer=utility__update__pb2.UtilityAck.FromString,
                )


class UtilityMessagingServicer(object):
    """Interface
    """

    def PublishUtility(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_UtilityMessagingServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'PublishUtility': grpc.unary_unary_rpc_method_handler(
                    servicer.PublishUtility,
                    request_deserializer=utility__update__pb2.UtilityMessage.FromString,
                    response_serializer=utility__update__pb2.UtilityAck.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Jingle.UtilityMessaging', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class UtilityMessaging(object):
    """Interface
    """

    @staticmethod
    def PublishUtility(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Jingle.UtilityMessaging/PublishUtility',
            utility__update__pb2.UtilityMessage.SerializeToString,
            utility__update__pb2.UtilityAck.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)