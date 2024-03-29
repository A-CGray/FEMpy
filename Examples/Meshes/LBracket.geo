// Gmsh project created on Mon Nov 14 20:29:56 2022
//+
Point(1) = {0, 1, 0, 1.0};
//+
Point(2) = {0, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {1, 0.4, 0, 1.0};
//+
Point(5) = {0.4, 0.4, 0, 1.0};
//+
Point(6) = {0.4, 1.0, 0, 1.0};
//+
Point(7) = {1.0, 0.36, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 7};
//+
Line(3) = {7, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 1};
//+
Line(7) = {1, 2};
//+
Curve Loop(1) = {6, 7, 1, 2, 3, 4, 5};
//+
Plane Surface(1) = {1};
//+
Physical Curve("FixedBC", 8) = {6};
//+
Physical Curve("LoadedEdge", 9) = {3};
//+
Physical Surface("surf", 10) = {1};
//+
Physical Curve(" FixedBC", 8) -= {6};
//+
Physical Curve(" LoadedEdge", 9) -= {3};
//+
MeshSize {5} = 0.01;
//+
MeshSize {5} = 0.01;
//+
MeshSize {2} = 0.1;
//+
MeshSize {2} = 10;
//+
MeshSize {2} = 100;
//+
MeshSize {2} = 1;
//+
MeshSize {2} = 0.1;
//+
MeshSize {2} = 1;
//+
MeshSize {1, 2, 3} = 1;
//+
MeshSize {6, 4} = 0.1;
//+
MeshSize {2} = 1;
//+
MeshSize {2} = 10;
//+
MeshSize {6, 5, 4} = 0.1;
//+
MeshSize {6, 5, 4} = 1;
//+
MeshSize {2} = 10;
//+
MeshSize {2} = 10;
//+
MeshSize {5, 4, 6} = 0.1;
