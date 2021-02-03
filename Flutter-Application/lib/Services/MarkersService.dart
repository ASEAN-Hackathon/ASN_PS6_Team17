import 'package:app2/Screens/MyCatchesScreen.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:app2/FishCard.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class MarkerService {
  List<Marker> getMarkers(List<FishCard> places,BitmapDescriptor myIcon,context) {
    List<Marker> markers = <Marker>[];
    places.forEach((place) {
      Marker marker = Marker(
          markerId: MarkerId(place.catchId.toString()),
          draggable: false,
          infoWindow: InfoWindow(
              title: place.catchId.toString(),
              snippet: place.weight.toString()+" kg",
              onTap: () => {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => MyDetailsScreen(place.catchesFish),
                      ),
                    )
                  }),
          position:
              LatLng(place.latitude,place.longitude),
          icon: myIcon);
      markers.add(marker);
    });
    return markers;
  }
}



// ignore: must_be_immutable
class MyDetailsScreen extends StatelessWidget {
  List<Catch> fishes;

  MyDetailsScreen(this.fishes);
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Nearby Catch Details!')), 
        body:
        Container(
            child: ListView.builder(
                padding: EdgeInsets.all(5.0),
                itemCount: fishes.length,
                itemBuilder: (BuildContext context, int index) {
                  return InkWell(
                    onTap: (){
                        // print(fishes[index]['location']);
                    },
                      child: Card(
                      color: shrinePink100,
              child: Column(
                children: [
                    ListTile(
                      leading: Icon(Icons.details),
                      title: Text(fishes[index].fishType,style: TextStyle(fontSize: 20),),
                      subtitle: Text(
                        "Cost: "+fishes[index].cost.toString(),
                        style: TextStyle(color: Colors.black.withOpacity(0.6)),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        "Quantity: "+ fishes[index].quantity.toString() + "\nWeight: " + fishes[index].weight.toString(),
                        style: TextStyle(color: Colors.black.withOpacity(0.6),fontSize: 18),
                      ),
                    ),
                    // (fishes[index]['imageAssociated'] == null) ? Image.asset('assets/images/fish.png') : Image.network(fishes[index]['imageAssociated']),
                    SizedBox(height:20),
                ],
              ),
            ),
                  );
                }),
          ));
  }

}