#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#define PI 3.141592653589
int maxn=200;
double g=10;
double dt=0.01;
typedef struct status{
    double t1;
    double t2;
    double dt1;
    double dt2;
}Status;
Status mul(Status a,double data){
    Status neww=a;
    neww.t1*=data;
    neww.t2*=data;
    neww.dt1*=data;
    neww.dt2*=data;
    return neww;
}
double reflect(double a){
    while(a<-PI){
        a+=2*PI;
    }
    while(a>PI){
        a-=2*PI;
    }
    return a;
}
double abss(double a){
    return a>0?a:-a;
}
double color(Status a){
    return abss(a.t1*PI+a.t2)/(PI*PI+PI);
}
Status add(Status a,Status b){
    Status neww=(Status){a.t1+b.t1,a.t2+b.t2,a.dt1+b.dt1,a.dt2+b.dt2};
    return neww;
}
double f1(Status a){
    return a.dt1;
}
double f2(Status a){
    return a.dt2;
}
double g1(Status a){
    double temp1=2-cos(a.t1-a.t2)*cos(a.t1-a.t2);
    double delta=a.t1-a.t2;
    double temp2=(a.dt1-a.dt2)*sin(delta)*(a.dt2-a.dt1*cos(delta))+g*(sin(a.t2)*cos(delta)-2*sin(a.t1));
    return temp2/temp1;
}
double g2(Status a){
    double temp1=2-cos(a.t1-a.t2)*cos(a.t1-a.t2);
    double delta=a.t1-a.t2;
    double temp2=(2*a.dt1-a.dt2*cos(delta))*(a.dt1-a.dt2)*sin(delta)-2*g*sin(a.t2)+2*g*sin(a.t1)*cos(delta);
    return temp2/temp1;
}
Status deltastatus(Status c,Status delta){
    Status current=(Status){c.t1+delta.t1,c.t2+delta.t2,c.dt1+delta.dt1,c.dt2+delta.dt2};
    Status next;
    next.t1=dt*f1(current);
    next.t2=dt*f2(current);
    next.dt1=dt*g1(current);
    next.dt2=dt*g2(current);
    return next;
}
Status calculate(Status current){
    Status delta=(Status){0.0,0.0,0.0,0.0};
    Status k1=deltastatus(current,delta);
    Status k2=deltastatus(current,mul(k1,0.5));
    Status k3=deltastatus(current,mul(k2,0.5));
    Status k4=deltastatus(current,k3);
    Status next=add(current,mul(add(k1,add(mul(k2,2),add(mul(k3,2),k4))),1.0/6.0));
    next.t1=reflect(next.t1);
    next.t2=reflect(next.t2);
    return next;
}
int maxframe=5000;
char filename[50];
int main(void){
    for(int i=0;i<maxn;i++){
        snprintf(filename,sizeof(filename),"data%d.txt",i);
        FILE* file=fopen(filename,"w");
        for(int j=0;j<maxn;j++){
            Status current=(Status){2.0*PI*(double)i/(double)maxn-PI,2.0*PI*(double)j/maxn-PI,0.0,0.0};
            for(int k=0;k<maxframe;k++){
                Status next=calculate(current);
                double temp=color(current);
                fprintf(file,"%lf ",temp);
                current=next;
            }
            fprintf(file,"\n");
        }
        fclose(file);
        printf("complete %d\n",i);
    }
    return 0;
}