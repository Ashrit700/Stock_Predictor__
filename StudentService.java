package com.example.demo.service;

import com.example.demo.model.Student;
import java.util.List;

public interface StudentService {
    Student save(Student student);
    List<Student> getAll();
    Student getById(Long id);
    Student update(Long id, Student student);
    String delete(Long id);
}
