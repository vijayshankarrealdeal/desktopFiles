#include<bits/stdc++.h>
using namespace std;

int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        int a,b;
        cin>>a>>b;
        string s;
        cin>>s;
        int x=0,y=0;
        for(int i=0;i<s.size();i++)
        {
            if(s[i]=='L')
            {
                x=x-1;
                y=y;
            }
            else if(s[i]=='R')
            {
                x=x+1;
                y=y;
            }
            else if(s[i]=='U')
            {
                x=x;
                y=y+1;
            }
            else if(s[i]=='D')
            {
                x=x;
                y=y-1;
            }
        }
        
        int dx=x;
        int dy=y;
        if(a*x<0 || b*y<0)
        cout<<"No"<<endl;
        else
        {
            if(a*x>=0 && b*y>=0)
            {
                if(x>a || y>b)
                cout<<"No"<<endl;
                else
                {
                    while(x<a || y<b)
                    {
                        x=x+dx;
                        y=y+dy;
                    }
                    if(x==a && y==b)
                    cout<<"Yes"<<endl;
                    else
                    cout<<"No"<<endl;
                }
            }
        }
    }
}