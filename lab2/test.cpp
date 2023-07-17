#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    float a = 5555.55;

    ofstream ofs;
    ofs.open("res.txt");
    if ( !ofs.is_open())
    {
        cout<<"aaaaaa"<<endl;
        return 1;
    }

    ofs << a << "\n";

    return 0;
}