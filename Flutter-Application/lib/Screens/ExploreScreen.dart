import 'package:app2/FishCard.dart';
import 'package:app2/Services/MarkersService.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:app2/Services/FishCardService.dart';
class ExploreScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MyApp1();
  }
}

class MyApp1 extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Explore Catches Nearby!')), body: GeolocationExample());
  }
}


class GeolocationExampleState extends State {
  Geolocator _geolocator;
  Position _position;
  double _positionLatitude = 13.03;
  GoogleMapController mapController;
  double _positionLongitude = 77.56;
  List<FishCard> fishCards = [];
  FishCard f1 = new FishCard();
  BitmapDescriptor icon;
  FishCard f2 = new FishCard();
  MarkerService markerService;
  var markers;

  Future<void> _gotoLocation(double lat, double long) async {
      mapController.animateCamera(CameraUpdate.newCameraPosition(CameraPosition(
        target: LatLng(lat, long),
        zoom:15,
        tilt: 50.0,
        bearing: 45.0,
      )));
    }
  final apiKey = "AIzaSyCILGP87TZPkXUobQfqDp9mkPA7IXnEGXU";
  void checkPermission() {
    _geolocator.checkGeolocationPermissionStatus().then((status) {
      print('status: $status');
    });
    _geolocator
        .checkGeolocationPermissionStatus(
            locationPermission: GeolocationPermission.locationAlways)
        .then((status) {
      print('always status: $status');
    });
    _geolocator.checkGeolocationPermissionStatus(
        locationPermission: GeolocationPermission.locationWhenInUse)
      ..then((status) {
        print('whenInUse status: $status');
      });
  }


Widget _buildContainer(List<FishCard> placeMarkers) {
    
    return Align(
      alignment: Alignment.bottomLeft,
      child: Container(
        height: 150.0,
        child: ListView(
          scrollDirection: Axis.horizontal,
          children: <Widget>[
            for (var place in placeMarkers)
              Padding(
                padding: const EdgeInsets.only(
                    right: 5, left: 5, top: 5, bottom: 5),
                child: _boxes(place),
              ),
          ],
        ),
      ),
    );
  }

  Widget _boxes(FishCard place) {
    return Container(
      child: new FittedBox(
        child: Material(
            color: Colors.white,
            elevation: 5.0,
            borderRadius: BorderRadius.circular(10.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: <Widget>[
                Container(
                  child: Padding(
                    padding: const EdgeInsets.all(2.0),
                    child: myDetailsContainer1(place),
                  ),
                ),
              ],
            )),
      ),
    );
  }

  Widget myDetailsContainer1(FishCard place) {
    return Material(
        child: InkWell(
      child: Container(
          height: MediaQuery.of(context).size.height / 3,
          width: 350.0,
          child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                Container(
                    child: Row(children: [
                  Text(
                    "Catch Discovered",
                    style: TextStyle(
                        color: Colors.lightBlue[600],
                        fontSize: 28.0,
                        fontWeight: FontWeight.bold),
                  ),
                  Spacer(),
                ])),
                Container(
                    child: Text(
                  "CatchId: "+place.catchId.toString(),
                  style: TextStyle(
                    color: Colors.blue[600],
                    fontSize: 24.0,
                  ),
                )),
                Container(
                    child: Text(
                  "Catch Weight: "+place.weight.toString() + " kg",
                  style: TextStyle(
                    color: Colors.blue[600],
                    fontSize: 24.0,
                  ),
                )),
                Container(
                    child: Text(
                  "Catch Date: "+place.date,
                  style: TextStyle(
                    color: Colors.blue[600],
                    fontSize: 24.0,
                  ),
                ))
                  ],
                )),
      onTap: () => _gotoLocation(
          place.latitude,place.longitude),
    ));
  }
  void updateLocation() async {
    try {
      Position newPosition = await Geolocator()
          .getCurrentPosition(desiredAccuracy: LocationAccuracy.high)
          .timeout(new Duration(seconds: 10));
      setState(() {
        _position = newPosition;
        _positionLatitude = _position.latitude;
        _positionLongitude = _position.longitude;
        print(_positionLatitude);
        print(_positionLongitude);
        _gotoLocation(_positionLatitude,_positionLongitude);
      });
    } catch (e) {
      print('Error: ${e.toString()}');
    }
  }

 void updatePlace(double lat, double long) async {
    List<FishCard> places = await FishCardService().getFishes(lat, long);
    setState(() {
      fishCards = places;
      updateMarkers(fishCards);
    });
  }


 void updateMarkers(List<FishCard> fishCards) async {
   debugPrint("hello");
   debugPrint(fishCards.toString());
      markers = (fishCards != null)
        ? markerService.getMarkers(
            fishCards,
            icon,
            context)
        : List<Marker>();
    markers.add(Marker(
        markerId: MarkerId('Base Location'),
        draggable: false,
        consumeTapEvents: false,
        icon: BitmapDescriptor.defaultMarker,
        position: LatLng(_positionLatitude, _positionLongitude)));
  }



  @override
  void initState() {
    super.initState();
      BitmapDescriptor.fromAssetImage(ImageConfiguration(size: Size(5,5)),
            'assets/icons/icon1.png')
        .then((onValue) {
      icon = onValue;
    });
    updateMarkers(fishCards);
    _geolocator = Geolocator();
    markerService = MarkerService();
    updateLocation();
    debugPrint(_positionLatitude.toString()+" _"+_positionLongitude.toString());
    updatePlace(_positionLatitude,_positionLongitude);
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: (_positionLatitude != null && markers != null && _positionLongitude != null && fishCards != null)
          ? Container(
              child: Column(
                children: <Widget>[
                  Stack(
                    children: <Widget>[
                      Container(
                        height: 2*MediaQuery.of(context).size.height/3,
                        width: MediaQuery.of(context).size.width,
                        child: GoogleMap(
                            initialCameraPosition: CameraPosition(
                                target: LatLng(
                                    _positionLatitude, _positionLongitude),
                                zoom: 14.0),
                            onMapCreated: (GoogleMapController controller) {
                              mapController = controller;
                            },
                            // gestureRecognizers: Set().add(Factory<PanGestureRecognizer>(
                            //       () => PanGestureRecognizer())),
                            markers: Set<Marker>.of(markers),
                            zoomGesturesEnabled: true,
                            myLocationButtonEnabled: false,
                            myLocationEnabled: true,
                            compassEnabled: false,
                            mapToolbarEnabled: true,
                            circles: Set.from([
                              Circle(
                                  circleId: CircleId('0'),
                                  center: LatLng(
                                      _positionLatitude, _positionLongitude),
                                  radius: 1250,
                                  strokeColor: Color(0x5DA9CAff),
                                  strokeWidth: 1,
                                  fillColor: Color(0x5DA9CAff)),
                              Circle(
                                  circleId: CircleId('1'),
                                  center: LatLng(
                                      _positionLatitude, _positionLongitude),
                                  radius: 2500,
                                  strokeColor: Color(0x3DA9CAff),
                                  strokeWidth: 1,
                                  fillColor: Color(0x3DA9CAff)),
                              Circle(
                                  circleId: CircleId('2'),
                                  center: LatLng(
                                      _positionLatitude, _positionLongitude),
                                  radius: 5000,
                                  strokeColor: Color(0x2DA9CAff),
                                  strokeWidth: 1,
                                  fillColor: Color(0x2DA9CAff))
                            ])),
                      ),
                      Padding(
                        padding: EdgeInsets.only(
                            top: 2 * MediaQuery.of(context).size.height / 3),
                        child: _buildContainer(fishCards),
                      )
                    ],
                  ),
                ],
              ),
            )
          : Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                ],
            )),
    );
  }
}

class GeolocationExample extends StatefulWidget {
  @override
  GeolocationExampleState createState() => new GeolocationExampleState();
}