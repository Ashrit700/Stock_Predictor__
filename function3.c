#include<stdio.h>
int sum(int a,int b){
int s=0;
s=a+b;
return s;
}
int sub(int a,int b){
    int s=0;
    s=a-b;
    return s;
}
int mul(int a,int b){
    int s=0;
    s=a*b;
    return s;
}
int div(int a,int b){
    int s=0;
    s=a/b;
    return s;
}
int main(){
    int a,b;
    scanf("%d %d",&a,&b);
    int Sum,Sub,Mul,Div;
    Sum=sum(a,b);
    Sub=sub(a,b);
    Mul=mul(a,b);
    Div=div(a,b);
    printf("%d\n %d\n %d\n %d\n",Sum,Sub,Mul,Div);
    return 0;
}
    
    
    
