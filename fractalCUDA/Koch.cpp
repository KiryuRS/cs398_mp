#include "Koch.h"



void KochCPU(list<Line*>& lines)
{
	list<Line*> newlines;
	list<Line*> TobeDeleteLine;

	for (auto itr = lines.begin(); itr != lines.end(); itr++)
	{
		double xl1 = (*itr)->x;
		double yl1 = (*itr)->y;
		double lenl1 = (*itr)->len / 3;
		double angl1 = (*itr)->ang;

		double xl2 = (*itr)->x + (cos((*itr)->ang*(PI / 180.0))*(*itr)->len / 1.5);
		double yl2 = (*itr)->y + (sin((*itr)->ang*(PI / 180.0))*(*itr)->len / 1.5);
		double lenl2 = (*itr)->len / 3;
		double angl2 = (*itr)->ang;

		double xl3 = (*itr)->x + (cos((*itr)->ang*(PI / 180.0))*(*itr)->len / 3.0);
		double yl3 = (*itr)->y + (sin((*itr)->ang*(PI / 180.0))*(*itr)->len / 3.0);
		double lenl3 = (*itr)->len / 3.0;
		double angl3 = (*itr)->ang- 300.0;

		double xl4 = (*itr)->x + (cos((*itr)->ang*(PI / 180.0))*((*itr)->len / 1.5));
		double yl4 = (*itr)->y + (sin((*itr)->ang*(PI / 180.0))*((*itr)->len / 1.5));
		double lenl4 = (*itr)->len / 3.0;
		double angl4 = (*itr)->ang- 240.0;

		xl4 = xl4 + cos(angl4*(PI / 180.0))*lenl4;
		yl4 = yl4 + sin(angl4*(PI / 180.0))*lenl4;
		angl4 -= 180.0;

		newlines.push_back(new Line(xl1, yl1, lenl1, angl1));
		newlines.push_back(new Line(xl2, yl2, lenl2, angl2));
		newlines.push_back(new Line(xl3, yl3, lenl3, angl3));
		newlines.push_back(new Line(xl4, yl4, lenl4, angl4));

		//...for deleting itself!
		TobeDeleteLine.push_back((*itr));
	}
	for (auto itr = newlines.begin(); itr != newlines.end(); itr++)
		lines.push_back((*itr)); //Adding new Line*(s)

	for (auto itr = TobeDeleteLine.begin(); itr != TobeDeleteLine.end(); itr++)
	{
		lines.remove((*itr)); //Deleting new Line*(s)
		delete (*itr);
	}
}

void RunKoch(uchar * data)
{
	std::list<Line*> lines;
	lines.push_back(new Line(412, 150, 312, 180.0));
	lines.push_back(new Line(100, 150, 312, 60.0));
	Line* line = new Line(412, 150, 312, 120);
	line->x += cos(line->ang * (PI / 180))*line->len;
	line->y += cos(line->ang * (PI / 180))*line->len;
	line->ang -= 180;
	lines.push_back(line);
	
	
	for (auto itr = lines.begin(); itr != lines.end(); itr++)
	{
		(*itr)->draw((*itr)->x, (*itr)->y, 0, data);
		
		
	}
	Sleep(2500);
	KochCPU(lines);
	
		
	
	for (auto itr = lines.begin(); itr != lines.end(); itr++)
		delete (*itr); //Deleting all lines at the end of a program
}
